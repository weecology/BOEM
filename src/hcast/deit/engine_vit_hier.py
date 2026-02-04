# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
from . import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    if len(args.nb_classes) == 3:
        for samples, targets, family_targets, mf_targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            family_targets = family_targets.to(device, non_blocking=True)
            mf_targets = mf_targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                samples, targets, family_targets, mf_targets = mixup_fn(samples, [targets, family_targets, mf_targets])
                
                
            if args.cosub:
                samples = torch.cat((samples,samples),dim=0)
                
            if args.bce_loss:
                targets = targets.gt(0.0).type(targets.dtype)
            
            with torch.cuda.amp.autocast():
                outputs, family_out, manu_out = model(samples)
                if not args.cosub:
                    loss_species = criterion(samples, outputs, targets)
                    loss_family = criterion(samples, family_out, family_targets)
                    loss_manufacturer = criterion(samples, manu_out, mf_targets)
                    loss = loss_species + loss_family + loss_manufacturer

                else:
                    outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                    loss = 0.25 * criterion(outputs[0], targets) 
                    loss = loss + 0.25 * criterion(outputs[1], targets) 
                    loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                    loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

            metric_logger.update(sp_loss=loss_species.item())
            metric_logger.update(fam_loss=loss_family.item())
            metric_logger.update(manu_loss=loss_manufacturer.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    else:
        for samples, targets, family_targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            family_targets = family_targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                samples, targets, family_targets = mixup_fn(samples, [targets, family_targets])
                
                
            if args.cosub:
                samples = torch.cat((samples,samples),dim=0)
                
            if args.bce_loss:
                targets = targets.gt(0.0).type(targets.dtype)
            
            with torch.cuda.amp.autocast():
                outputs, family_out = model(samples)
                if not args.cosub:
                    loss_species = criterion(samples, outputs, targets)
                    loss_family = criterion(samples, family_out, family_targets)
                    loss = loss_species + loss_family 

                else:
                    outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                    loss = 0.25 * criterion(outputs[0], targets) 
                    loss = loss + 0.25 * criterion(outputs[1], targets) 
                    loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                    loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

            metric_logger.update(sp_loss=loss_species.item())
            metric_logger.update(fam_loss=loss_family.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, n_classes=3):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if n_classes == 3:
        for images, target, family_targets, mf_targets in metric_logger.log_every(data_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            family_targets = family_targets.to(device, non_blocking=True)
            mf_targets = mf_targets.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output, family_out, manu_out = model(images)
                loss_species = criterion(output, target)
                loss_family = criterion(family_out, family_targets)
                loss_manufacturer = criterion(manu_out, mf_targets)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            family_acc1, family_acc5 = accuracy(family_out, family_targets, topk=(1, 5))
            manu_acc1, manu_acc5 = accuracy(manu_out, mf_targets, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(sploss=loss_species.item())
            metric_logger.update(famloss=loss_family.item())
            metric_logger.update(manuloss=loss_manufacturer.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters['family_acc1'].update(family_acc1.item(), n=batch_size)
            metric_logger.meters['manu_acc1'].update(manu_acc1.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} family@1 {familytop1.global_avg:.3f}' 
            ' manu@1 {manutop1.global_avg:.3f} sploss {losses.global_avg:.3f} fmloss {fmlosses.global_avg:.3f} mfloss {mflosses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.sploss, fmlosses=metric_logger.famloss, mflosses=metric_logger.manuloss,
                    familytop1=metric_logger.family_acc1, manutop1=metric_logger.manu_acc1))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    else:
        for images, target, family_targets in metric_logger.log_every(data_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            family_targets = family_targets.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output, family_out = model(images)
                loss_species = criterion(output, target)
                loss_family = criterion(family_out, family_targets)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            family_acc1, family_acc5 = accuracy(family_out, family_targets, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(sploss=loss_species.item())
            metric_logger.update(famloss=loss_family.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters['family_acc1'].update(family_acc1.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} family@1 {familytop1.global_avg:.3f}' 
            ' sploss {losses.global_avg:.3f} fmloss {fmlosses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.sploss, fmlosses=metric_logger.famloss, 
                    familytop1=metric_logger.family_acc1))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
