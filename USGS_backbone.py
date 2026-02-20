from deepforest import main
import pandas as pd
import os
from pytorch_lightning.loggers import CometLogger
import torch
import argparse
import tempfile
from deepforest import visualize
from deepforest.utilities import read_file

# Parse arguments
parser = argparse.ArgumentParser(description="Train DeepForest model")
parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
parser.add_argument("--workers", type=int, default=5, help="Number of workers for data loading")
parser.add_argument(
    "--max-empty-fraction",
    type=float,
    default=None,
    help="Cap proportion of empty images in train (e.g. 0.3 = max 30%%). Subsamples empty images and saves train_max_empty_<frac>.csv.",
)
args = parser.parse_args()

# Use parsed arguments
batch_size = args.batch_size
workers = args.workers

savedir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
root_dir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"

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
m = main.deepforest()
m.load_model("weecology/deepforest-bird")
m.label_dict = {"Object":0}
m.numeric_to_label_dict = {0:"Object"}

m.config["train"]["csv_file"] = train_csv_path
m.config["train"]["root_dir"] = root_dir
m.config["train"]["fast_dev_run"] = False
m.config["validation"]["csv_file"] = os.path.join(savedir,"test.csv")
m.config["validation"]["root_dir"] = root_dir
m.config["batch_size"] = batch_size
m.config["train"]["epochs"] = 23
m.config["workers"] = workers
m.config["validation"]["val_accuracy_interval"] = 2
m.config["train"]["scheduler"]["params"]["eps"]  = 0
m.config["train"]["lr"] = 0.001
m.config["train"]["scheduler"]["params"]["patience"] = 3

comet_logger = CometLogger(project_name="BOEM", workspace="bw4sz")
comet_logger.experiment.add_tag("detection")

# Log the training and test sets
comet_logger.experiment.log_table("train.csv", train)
comet_logger.experiment.log_table("test.csv", test)

# Pytorch lightning save checkpoint
#simple_profiler = SimpleProfiler(dirpath=os.path.join(tmpdir,"profiler"), filename="profiler.txt", extended=True)

# Log the devices
devices = torch.cuda.device_count()
comet_logger.experiment.log_parameter("devices", devices)
if args.max_empty_fraction is not None:
    comet_logger.experiment.log_parameter("max_empty_fraction", args.max_empty_fraction)
comet_logger.experiment.log_parameter("workers", m.config["workers"])
comet_logger.experiment.log_parameter("batch_size", m.config["batch_size"])

# Log data sizes
comet_logger.experiment.log_parameter("train_size", train.shape[0])
comet_logger.experiment.log_parameter("test_size", test.shape[0])

m.create_trainer(logger=comet_logger, accelerator="gpu", strategy="ddp", num_nodes=1, devices=devices, fast_dev_run=False)

# # Create a temporary directory for saving visualizations
# with tempfile.TemporaryDirectory() as tmpdir:
#     # Filter non-empty train annotations
#     non_empty_train = train[~train.empty_image]
#     n_train = min(5, non_empty_train.shape[0])
#     for img_path in non_empty_train.image_path.sample(n=n_train).unique():
#         ann = non_empty_train[non_empty_train.image_path == img_path]
#         ann.root_dir = savedir
#         ann = read_file(ann, root_dir=m.config["validation"]["root_dir"])
#         short_name = os.path.basename(img_path)
#         visualize.plot_annotations(ann, root_dir=ann.root_dir, savedir=tmpdir)
#         comet_logger.experiment.log_image(
#             os.path.join(tmpdir, short_name),
#             metadata={"name": short_name, "context": "detection_train"}
#         )

#     # Filter non-empty test annotations
#     non_empty_test = test[~test.empty_image]
#     n_test = min(5, non_empty_test.shape[0])
#     for img_path in non_empty_test.image_path.sample(n=n_test).unique():
#         ann = non_empty_test[non_empty_test.image_path == img_path]
#         ann.root_dir = savedir
#         ann = read_file(ann, root_dir=m.config["validation"]["root_dir"])
#         short_name = os.path.basename(img_path)
#         visualize.plot_annotations(ann, root_dir=ann.root_dir, savedir=tmpdir)
#         comet_logger.experiment.log_image(
#             os.path.join(tmpdir, short_name),
#             metadata={"name": short_name, "context": "detection_validation"}
#         )
results = m.evaluate(
    csv_file = m.config["validation"]["csv_file"],
    root_dir = m.config["validation"]["root_dir"])

print(results)

m.trainer.fit(m)

# Save the model
m.trainer.save_checkpoint("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/checkpoints/{}.pl".format(comet_logger.experiment.id))

results = m.evaluate(
    csv_file = m.config["validation"]["csv_file"],
    root_dir = m.config["validation"]["root_dir"])

print(results)
# Log the evaluation results
comet_logger.experiment.log_metric("box_precision_after", results["box_precision"])
comet_logger.experiment.log_metric("box_recall_after", results["box_recall"])

# Zero-shot evaluation on held-out flights (generalization to unseen flights)
zeroshot_csv = os.path.join(savedir, "zero_shot.csv")
if os.path.isfile(zeroshot_csv):
    zeroshot_df = pd.read_csv(zeroshot_csv)
    comet_logger.experiment.log_parameter("zero_shot_size", len(zeroshot_df))
    zeroshot_results = m.evaluate(
        csv_file=zeroshot_csv,
        root_dir=m.config["validation"]["root_dir"],
    )
    print("Zero-shot (held-out flights) results:", zeroshot_results)
    comet_logger.experiment.log_metric("zero_shot_box_precision", zeroshot_results["box_precision"])
    comet_logger.experiment.log_metric("zero_shot_box_recall", zeroshot_results["box_recall"])
else:
    print("No zero_shot.csv found; skipping zero-shot metrics.")

# Gather the number of steps taken from all GPUs
global_steps = torch.tensor(m.trainer.global_step, dtype=torch.int32, device=m.device)
comet_logger.experiment.log_metric("global_steps", global_steps)
