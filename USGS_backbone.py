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
args = parser.parse_args()

# Use parsed arguments
batch_size = args.batch_size
workers = args.workers

savedir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
train = pd.read_csv(os.path.join(savedir,"train.csv"))
test = pd.read_csv(os.path.join(savedir,"test.csv"))

# Print the number of empty images in train and test sets
print("Number of empty images in train set:", train[train.empty_image].shape[0])
print("Number of empty images in test set:", test[test.empty_image].shape[0])

# Initalize Deepforest model
m = main.deepforest()
m.load_model("weecology/deepforest-bird")
m.label_dict = {"Object":0}
m.numeric_to_label_dict = {0:"Object"}

m.config["train"]["csv_file"] = os.path.join(savedir,"train.csv")
m.config["train"]["root_dir"] = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
m.config["train"]["fast_dev_run"] = False
m.config["validation"]["csv_file"] = os.path.join(savedir,"test.csv")
m.config["validation"]["root_dir"] = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
m.config["batch_size"] = batch_size
m.config["train"]["epochs"] = 25
m.config["workers"] = workers
m.config["validation"]["val_accuracy_interval"] = 5
m.config["train"]["scheduler"]["params"]["eps"]  = 0
m.config["train"]["lr"] = 0.001

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
    root_dir = m.config["validation"]["root_dir"],
    batch_size=36)

print(results)

m.trainer.fit(m)

# Save the model
m.trainer.save_checkpoint("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/checkpoints/{}.pl".format(comet_logger.experiment.id))

results = m.evaluate(
    csv_file = m.config["validation"]["csv_file"],
    root_dir = m.config["validation"]["root_dir"],
    batch_size=36)

print(results)
# Log the evaluation results
comet_logger.experiment.log_metric("box_precision", results["box_precision"])
comet_logger.experiment.log_metric("box_recall", results["box_recall"])

# Gather the number of steps taken from all GPUs
global_steps = torch.tensor(m.trainer.global_step, dtype=torch.int32, device=m.device)
comet_logger.experiment.log_metric("global_steps", global_steps)
