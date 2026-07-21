from deepforest import main
import pandas as pd
import os
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Callback
import torch
import argparse
import tempfile
from contextlib import nullcontext
from deepforest import visualize
from deepforest import augmentations as df_aug
from deepforest.callbacks import ImagesCallback
from deepforest.utilities import read_file
from omegaconf import OmegaConf
import kornia.augmentation as K

# Register RandomAffine so it can be referenced from the augmentations config.
df_aug._SUPPORTED_TRANSFORMS["RandomAffine"] = (
    K.RandomAffine, {"degrees": 0, "scale": (0.9, 1.1), "p": 0.2}
)

# Parse arguments
parser = argparse.ArgumentParser(description="Train DeepForest model")
parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
parser.add_argument("--workers", type=int, default=5, help="Number of workers for data loading")
parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
parser.add_argument("--epochs", type=int, default=30, help="Max epochs for training")
parser.add_argument(
    "--early-stopping-patience",
    type=int,
    default=None,
    help="If set, enable EarlyStopping on val_classification with this patience (epochs).",
)
parser.add_argument(
    "--n-log-images",
    type=int,
    default=20,
    help="Number of validation/zero-shot images to log with predictions.",
)
parser.add_argument(
    "--max-empty-fraction",
    type=float,
    default=None,
    help="Cap proportion of empty images in train (e.g. 0.3 = max 30%%). Subsamples empty images and saves train_max_empty_<frac>.csv.",
)
parser.add_argument(
    "--max-test-empty-fraction",
    type=float,
    default=None,
    help="Cap proportion of empty images in the validation/test set (e.g. 0.25 = max 25%%).",
)
parser.add_argument(
    "--positive-batch-fraction",
    type=float,
    default=None,
    help="Fraction of each train batch reserved for annotated images "
         "(rest are hard negatives). None = uniform shuffle (DeepForest default). "
         "Requires the balanced-batch DeepForest branch.",
)
parser.add_argument(
    "--exp-name",
    type=str,
    default=None,
    help="Comet experiment name. Falls back to EXP_NAME env var, then an auto-derived name.",
)
args = parser.parse_args()

exp_name = args.exp_name or os.environ.get("EXP_NAME")
if not exp_name:
    parts = ["detection"]
    if args.positive_batch_fraction is not None:
        parts.append(f"posfrac{args.positive_batch_fraction}")
    exp_name = "_".join(parts)

# Use parsed arguments
batch_size = args.batch_size
workers = args.workers

savedir = "/blue/ewhite/b.weinstein/BOEM/training/crops"
root_dir = "/blue/ewhite/b.weinstein/BOEM/training/crops"

train = pd.read_csv(os.path.join(savedir, "train.csv"), low_memory=False)
test = pd.read_csv(os.path.join(savedir, "test.csv"), low_memory=False)

# Optional: limit proportion of empty images in train
train_csv_path = os.path.join(savedir, "train.csv")
if args.max_empty_fraction is not None:
    if "empty_image" not in train.columns:
        train["empty_image"] = (
            (train["xmin"] == 0)
            & (train["xmax"] == 0)
            & (train["ymin"] == 0)
            & (train["ymax"] == 0)
        )
    per_image_empty = train.groupby("image_path")["empty_image"].all()
    empty_images = set(per_image_empty[per_image_empty].index)
    with_object_images = set(per_image_empty[~per_image_empty].index)
    n_with = len(with_object_images)
    max_empty = int(n_with * args.max_empty_fraction / (1 - args.max_empty_fraction)) if args.max_empty_fraction < 1.0 else len(empty_images)
    keep_empty = set(pd.Series(list(empty_images)).sample(n=min(max_empty, len(empty_images)), random_state=42).values)
    keep_images = with_object_images | keep_empty
    train = train[train["image_path"].isin(keep_images)].copy()
    frac_str = f"{args.max_empty_fraction:.2f}".replace(".", "_")
    train_csv_path = os.path.join(savedir, f"train_max_empty_{frac_str}.csv")
    train.to_csv(train_csv_path, index=False)
    print(f"Limited empty images to {args.max_empty_fraction:.0%}; saved {len(train)} rows to {train_csv_path}")

# Optional: limit proportion of empty images in validation/test set
test_csv_path = os.path.join(savedir, "test.csv")
if args.max_test_empty_fraction is not None:
    if "empty_image" not in test.columns:
        test["empty_image"] = (
            (test["xmin"] == 0)
            & (test["xmax"] == 0)
            & (test["ymin"] == 0)
            & (test["ymax"] == 0)
        )
    per_image_empty_test = test.groupby("image_path")["empty_image"].all()
    empty_test_images = set(per_image_empty_test[per_image_empty_test].index)
    with_object_test_images = set(per_image_empty_test[~per_image_empty_test].index)
    n_with_test = len(with_object_test_images)
    max_empty_test = int(n_with_test * args.max_test_empty_fraction / (1 - args.max_test_empty_fraction)) if args.max_test_empty_fraction < 1.0 else len(empty_test_images)
    keep_empty_test = set(pd.Series(list(empty_test_images)).sample(n=min(max_empty_test, len(empty_test_images)), random_state=42).values)
    keep_test_images = with_object_test_images | keep_empty_test
    test = test[test["image_path"].isin(keep_test_images)].copy()
    frac_str_test = f"{args.max_test_empty_fraction:.2f}".replace(".", "_")
    test_csv_path = os.path.join(savedir, f"test_max_empty_{frac_str_test}.csv")
    test.to_csv(test_csv_path, index=False)
    print(f"Limited test empty images to {args.max_test_empty_fraction:.0%}; saved {len(test)} rows to {test_csv_path}")

# Print the number of empty images in train and test sets
if "empty_image" not in train.columns:
    train["empty_image"] = (
        (train["xmin"] == 0)
        & (train["xmax"] == 0)
        & (train["ymin"] == 0)
        & (train["ymax"] == 0)
    )
if "empty_image" not in test.columns:
    test["empty_image"] = (
        (test["xmin"] == 0)
        & (test["xmax"] == 0)
        & (test["ymin"] == 0)
        & (test["ymax"] == 0)
    )
print("Number of empty images in train set:", train["empty_image"].sum())
print("Number of empty images in test set:", test["empty_image"].sum())

# Initalize Deepforest model
# score_thresh must be set via config_args here: it is baked into the torchvision
# RetinaNet at model construction. Assigning m.config["score_thresh"] after this
# point updates only the dict, not the live model.
m = main.deepforest(config_args={"score_thresh": 0.4})
m.load_model("weecology/deepforest-bird")
m.label_dict = {"Object":0}
m.numeric_to_label_dict = {0:"Object"}

m.config["train"]["csv_file"] = train_csv_path
m.config["train"]["root_dir"] = root_dir
m.config["train"]["fast_dev_run"] = False
m.config["validation"]["csv_file"] = test_csv_path
m.config["validation"]["root_dir"] = root_dir
m.config["batch_size"] = batch_size
m.config["train"]["epochs"] = args.epochs
m.config["workers"] = workers
m.config["validation"]["val_accuracy_interval"] = 3
# Skip per-row PIL.Image.open()-based coordinate check. Our bboxes are clean
# (verified via prepare_USGS bucket-split), and at ~1M annotation rows the
# check takes hours of pure I/O on Lustre with no findings.
m.config["train"]["validate_coordinates"] = False
m.config["validation"]["validate_coordinates"] = False
m.config["train"]["scheduler"]["type"] = "ReduceLROnPlateau"
m.config["train"]["scheduler"]["params"]["mode"] = "min"
m.config["train"]["scheduler"]["params"]["patience"] = 3
m.config["train"]["scheduler"]["params"]["factor"] = 0.5
m.config["train"]["scheduler"]["params"]["eps"] = 0
m.config["validation"]["lr_plateau_target"] = "val_classification"
m.config["train"]["lr"] = args.lr
# TF32 matmul for B200 speedup. bf16-mixed is disabled because DeepForest's
# format_boxes calls .numpy() on prediction tensors and numpy doesn't support
# bfloat16, which makes validation crash. Re-enable once upstream casts to float.
m.config["matmul_precision"] = "high"
if args.positive_batch_fraction is not None:
    m.config = OmegaConf.merge(
        m.config,
        {"train": {"positive_batch_fraction": args.positive_batch_fraction}},
    )

# Augmentations (kornia-based in DeepForest >=2.1). Aligned with the bird detector's
selected_augs = []
selected_augs.append({"Rotate": {"degrees": 15, "p": 0.5}})
selected_augs.append({"HorizontalFlip": {"p": 0.5}})
selected_augs.append({"VerticalFlip": {"p": 0.3}})
selected_augs.append({"RandomAffine": {"degrees": 0, "scale": (0.9, 1.1), "p": 0.3}})

m.config = OmegaConf.merge(m.config, {"train": {"augmentations": selected_augs}})

comet_logger = CometLogger(project_name="BOEM", workspace="bw4sz")
comet_logger.experiment.add_tag("detection")
comet_logger.experiment.set_name(exp_name)
comet_logger.experiment.log_parameter("exp_name", exp_name)

# Log the training and test sets
#comet_logger.experiment.log_table("train.csv", train)
#comet_logger.experiment.log_table("test.csv", test)

# Log the devices
devices = torch.cuda.device_count()
comet_logger.experiment.log_parameter("devices", devices)
if args.max_empty_fraction is not None:
    comet_logger.experiment.log_parameter("max_empty_fraction", args.max_empty_fraction)
if args.max_test_empty_fraction is not None:
    comet_logger.experiment.log_parameter("max_test_empty_fraction", args.max_test_empty_fraction)
comet_logger.experiment.log_parameter("workers", m.config["workers"])
comet_logger.experiment.log_parameter("batch_size", m.config["batch_size"])
comet_logger.experiment.log_parameter("lr", m.config["train"]["lr"])
comet_logger.experiment.log_parameter("epochs", m.config["train"]["epochs"])
if args.early_stopping_patience is not None:
    comet_logger.experiment.log_parameter(
        "early_stopping_patience", args.early_stopping_patience
    )
if args.positive_batch_fraction is not None:
    comet_logger.experiment.log_parameter(
        "positive_batch_fraction", args.positive_batch_fraction
    )
    comet_logger.experiment.add_tag("balanced_batches")

# Log data sizes
comet_logger.experiment.log_parameter("train_size", train.shape[0])
comet_logger.experiment.log_parameter("test_size", test.shape[0])


def comet_context(name):
    """DDP-safe Comet logging context.

    On non-zero ranks PyTorch-Lightning hands back a dummy experiment whose
    context_manager() returns None (not a context manager), which crashes a bare
    ``with comet_logger.experiment.context_manager(...)``. Fall back to a no-op so
    every rank still executes the collective validate()/fit() calls in the body.
    """
    cm = comet_logger.experiment.context_manager(name)
    return cm if cm is not None else nullcontext()


# Run id for checkpoint paths. Real Comet experiment id on rank 0; on non-zero DDP
# ranks experiment.id is a dummy (not a str), so fall back to the SLURM job id so all
# workers agree on a clean directory name (only rank 0 actually writes checkpoints).
_run_id = comet_logger.experiment.id
if not isinstance(_run_id, str):
    _run_id = os.environ.get("SLURM_JOB_ID", "local")


def _log_prediction_images(model, ann_df, image_root, context, n_images, tmpdir):
    """Run predict_image on n sample non-empty images and log overlays to Comet."""
    if ann_df is None or ann_df.empty:
        return
    if "empty_image" in ann_df.columns:
        candidates = ann_df[~ann_df["empty_image"]]
    else:
        candidates = ann_df
    if candidates.empty:
        return
    image_paths = candidates["image_path"].drop_duplicates()
    image_paths = image_paths.sample(n=min(n_images, len(image_paths)), random_state=42)
    for image_path in image_paths:
        full_path = os.path.join(image_root, image_path)
        if not os.path.exists(full_path):
            continue
        prediction = model.predict_image(path=full_path)
        if prediction is None or prediction.empty:
            continue
        gt = candidates[candidates["image_path"] == image_path]
        try:
            visualize.plot_results(
                prediction,
                ground_truth=gt if not gt.empty else None,
                image=full_path,
                savedir=tmpdir,
                show=False,
            )
        except Exception as e:
            print(f"Failed to plot {image_path}: {e}")
            continue
        basename = os.path.splitext(os.path.basename(image_path))[0]
        # plot_results saves as <basename>.png by default
        saved = os.path.join(tmpdir, f"{basename}.png")
        if not os.path.exists(saved):
            # fall back to any file containing the stem
            matches = [
                f for f in os.listdir(tmpdir)
                if f.startswith(basename) and f.endswith(".png")
            ]
            saved = os.path.join(tmpdir, matches[0]) if matches else None
        if saved and os.path.exists(saved):
            comet_logger.experiment.log_image(
                saved,
                name=f"{context}_{os.path.basename(image_path)}",
                metadata={"context": context, "image_path": image_path},
            )


# Pre-training zero-shot evaluation: metrics + sample prediction images on held-out flights
zeroshot_csv = os.path.join(savedir, "zero_shot.csv")
zero_shot_df = pd.read_csv(zeroshot_csv, low_memory=False) if os.path.exists(zeroshot_csv) else None
if zero_shot_df is not None and "empty_image" not in zero_shot_df.columns:
    zero_shot_df["empty_image"] = (
        (zero_shot_df["xmin"] == 0)
        & (zero_shot_df["xmax"] == 0)
        & (zero_shot_df["ymin"] == 0)
        & (zero_shot_df["ymax"] == 0)
    )

with comet_context("zero_shot_pretraining"):
    m.config["validation"]["csv_file"] = zeroshot_csv
    m.config["validation"]["root_dir"] = savedir
    # Force interval=1 so the one-shot validate() at epoch 0 actually computes
    # box metrics (DeepForest gates _compute_epoch_metrics on (epoch+1) % interval == 0).
    _saved_val_interval = m.config["validation"]["val_accuracy_interval"]
    m.config["validation"]["val_accuracy_interval"] = 1
    _zs_strategy = "ddp" if devices > 1 else "auto"
    m.create_trainer(
        logger=comet_logger, accelerator="gpu", strategy=_zs_strategy,
        num_nodes=1, devices=devices, fast_dev_run=False,
    )
    pre_zero_shot_results = m.trainer.validate(m)
    m.config["validation"]["val_accuracy_interval"] = _saved_val_interval
    pre_zero_shot_metrics = pre_zero_shot_results[0] if pre_zero_shot_results else {}
    print("\nZero-shot evaluation (pre-training):")
    print(f"  Box Precision: {pre_zero_shot_metrics.get('box_precision', 'N/A')}")
    print(f"  Box Recall: {pre_zero_shot_metrics.get('box_recall', 'N/A')}")
    print(f"  Empty Frame Accuracy: {pre_zero_shot_metrics.get('empty_frame_accuracy', 'N/A')}")
    if zero_shot_df is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _log_prediction_images(
                m, zero_shot_df, savedir,
                context="zero_shot_pretraining",
                n_images=args.n_log_images,
                tmpdir=tmpdir,
            )

# Restore validation config for training
m.config["validation"]["csv_file"] = test_csv_path
m.config["validation"]["root_dir"] = root_dir

# Optional EarlyStopping on val_classification (loss; lower is better)
callbacks = []
if args.early_stopping_patience is not None:
    callbacks.append(
        EarlyStopping(
            monitor="val_classification",
            mode="min",
            patience=args.early_stopping_patience,
            verbose=True,
        )
    )

images_save_dir = tempfile.mkdtemp(prefix="deepforest_images_")
callbacks.append(
    ImagesCallback(
        save_dir=images_save_dir,
        prediction_samples=5,
        dataset_samples=5,
        every_n_epochs=5,
        select_random=True,
    )
)

checkpoint_dir = f"/blue/ewhite/b.weinstein/BOEM/training/checkpoints/{_run_id}"
os.makedirs(checkpoint_dir, exist_ok=True)
callbacks.append(
    ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch:02d}-val_cls{val_classification:.4f}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )
)

# Single-GPU: "auto" avoids the DDP wrapper overhead that "ddp" pays even at nranks=1.
strategy = "ddp" if devices > 1 else "auto"
m.create_trainer(
    logger=comet_logger, accelerator="gpu", strategy=strategy,
    num_nodes=1, devices=devices, fast_dev_run=False,
    callbacks=callbacks,
)

m.trainer.fit(m)

# Save the model
m.trainer.save_checkpoint("/blue/ewhite/b.weinstein/BOEM/training/checkpoints/{}.pl".format(_run_id))

# Post-training validation prediction images (in-distribution errors)
with comet_context("validation_predictions"):
    with tempfile.TemporaryDirectory() as tmpdir:
        _log_prediction_images(
            m, test, root_dir,
            context="validation_post_training",
            n_images=args.n_log_images,
            tmpdir=tmpdir,
        )

# Zero-shot evaluation on held-out flights (generalization to unseen flights)
with comet_context("zero_shot_evaluation"):
    m.config["validation"]["csv_file"] = zeroshot_csv
    m.config["validation"]["root_dir"] = savedir
    _saved_val_interval = m.config["validation"]["val_accuracy_interval"]
    m.config["validation"]["val_accuracy_interval"] = 1
    _zs_strategy = "ddp" if devices > 1 else "auto"
    m.create_trainer(logger=comet_logger, accelerator="gpu", strategy=_zs_strategy, num_nodes=1, devices=devices, fast_dev_run=False)
    zero_shot_results = m.trainer.validate(m)
    m.config["validation"]["val_accuracy_interval"] = _saved_val_interval
    zero_shot_metrics = zero_shot_results[0] if zero_shot_results else {}
    print("\nZero-shot evaluation results (post-training):")
    print(f"  Box Precision: {zero_shot_metrics.get('box_precision', 'N/A')}")
    print(f"  Box Recall: {zero_shot_metrics.get('box_recall', 'N/A')}")
    print(f"  Empty Frame Accuracy: {zero_shot_metrics.get('empty_frame_accuracy', 'N/A')}")
    if zero_shot_df is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _log_prediction_images(
                m, zero_shot_df, savedir,
                context="zero_shot_post_training",
                n_images=args.n_log_images,
                tmpdir=tmpdir,
            )
