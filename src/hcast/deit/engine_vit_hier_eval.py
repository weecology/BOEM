# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""Detailed evaluation with FPA and TICE metrics for hierarchical ViT."""
import csv
import json
from typing import Optional

import torch
from timm.utils import accuracy

from . import utils


@torch.no_grad()
def evaluate_detail(data_loader, model, device, filename, n_classes=3, dataset='USGS', breeds_sort=None):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    tree_path = None
    if 'USGS' in dataset:
        tree_path = 'data/usgs_paths.json'

    trees = json.load(open(tree_path)) if tree_path else None

    model.eval()
    results = []
    tice_cnt = 0
    fpa_cnt = 0
    total_cnt = 0

    if n_classes == 3:
        results.append(['family_gt', 'family_pred', 'genus_gt', 'genus_pred', 'species_gt', 'species_pred'])
        for images, target, genus_targets, family_targets in metric_logger.log_every(data_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            genus_targets = genus_targets.to(device, non_blocking=True)
            family_targets = family_targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output, genus_out, family_out = model(images)
                loss_species = criterion(output, target)
                loss_genus = criterion(genus_out, genus_targets)
                loss_family = criterion(family_out, family_targets)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            genus_acc1, _ = accuracy(genus_out, genus_targets, topk=(1, 5))
            family_acc1, _ = accuracy(family_out, family_targets, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(sploss=loss_species.item())
            metric_logger.update(genusloss=loss_genus.item())
            metric_logger.update(famloss=loss_family.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters['genus_acc1'].update(genus_acc1.item(), n=batch_size)
            metric_logger.meters['family_acc1'].update(family_acc1.item(), n=batch_size)

            _, pred = torch.max(output, 1)
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()

            _, genus_pred = torch.max(genus_out, 1)
            genus_pred = genus_pred.cpu().numpy()
            genus_targets = genus_targets.cpu().numpy()

            _, family_pred = torch.max(family_out, 1)
            family_pred = family_pred.cpu().numpy()
            family_targets = family_targets.cpu().numpy()

            total_cnt += batch_size
            for i in range(batch_size):
                results.append([family_targets[i], family_pred[i], genus_targets[i], genus_pred[i], target[i], pred[i]])
                if pred[i] == target[i] and genus_pred[i] == genus_targets[i] and family_pred[i] == family_targets[i]:
                    fpa_cnt += 1
                if trees is not None:
                    tice_results = [family_pred[i], genus_pred[i], pred[i]]
                    if tice_results in trees:
                        tice_cnt += 1

        metric_logger.synchronize_between_processes()
        print(
            '* Species@1 {top1.global_avg:.3f} Species@5 {top5.global_avg:.3f} '
            'Genus@1 {genus.global_avg:.3f} Family@1 {family.global_avg:.3f} '
            'sploss {sl.global_avg:.3f} genusloss {gl.global_avg:.3f} famloss {fl.global_avg:.3f}'
            .format(
                top1=metric_logger.acc1, top5=metric_logger.acc5,
                genus=metric_logger.genus_acc1, family=metric_logger.family_acc1,
                sl=metric_logger.sploss, gl=metric_logger.genusloss, fl=metric_logger.famloss,
            )
        )

    else:
        results.append(['genus_gt', 'genus_pred', 'species_gt', 'species_pred'])
        for images, target, genus_targets in metric_logger.log_every(data_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            genus_targets = genus_targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output, genus_out = model(images)
                loss_species = criterion(output, target)
                loss_genus = criterion(genus_out, genus_targets)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            genus_acc1, _ = accuracy(genus_out, genus_targets, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(sploss=loss_species.item())
            metric_logger.update(genusloss=loss_genus.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters['genus_acc1'].update(genus_acc1.item(), n=batch_size)

            _, pred = torch.max(output, 1)
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()

            _, genus_pred = torch.max(genus_out, 1)
            genus_pred = genus_pred.cpu().numpy()
            genus_targets = genus_targets.cpu().numpy()

            total_cnt += batch_size
            for i in range(batch_size):
                results.append([genus_targets[i], genus_pred[i], target[i], pred[i]])
                if pred[i] == target[i] and genus_pred[i] == genus_targets[i]:
                    fpa_cnt += 1
                if trees is not None:
                    tice_results = [pred[i], genus_pred[i]]
                    if tice_results in trees:
                        tice_cnt += 1

        metric_logger.synchronize_between_processes()
        print(
            '* Species@1 {top1.global_avg:.3f} Species@5 {top5.global_avg:.3f} '
            'Genus@1 {genus.global_avg:.3f} '
            'sploss {sl.global_avg:.3f} genusloss {gl.global_avg:.3f}'
            .format(
                top1=metric_logger.acc1, top5=metric_logger.acc5,
                genus=metric_logger.genus_acc1,
                sl=metric_logger.sploss, gl=metric_logger.genusloss,
            )
        )

    print(f"FPA: {(fpa_cnt / total_cnt) * 100:.3f}% | TICE: {((total_cnt - tice_cnt) / total_cnt) * 100:.3f}% ")

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerows(results)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
