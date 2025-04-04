from deepforest import main
import pandas as pd
import os
import comet_ml
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.profilers.simple import SimpleProfiler
import torch
import argparse
from deepforest.callbacks import images_callback

# Parse arguments
parser = argparse.ArgumentParser(description="Train DeepForest model")
parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
parser.add_argument("--workers", type=int, default=16, help="Number of workers for data loading")
args = parser.parse_args()

# Use parsed arguments
batch_size = args.batch_size
workers = args.workers

savedir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
train = pd.read_csv(os.path.join(savedir,"train.csv"))
test = pd.read_csv(os.path.join(savedir,"test.csv"))

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
m.config["train"]["epochs"] = 50
m.config["workers"] = workers
m.config["validation"]["val_accuracy_interval"] = 20
m.config["train"]["scheduler"]["params"]["eps"]  = 0
m.config["train"]["lr"] = 0.0005

comet_logger = CometLogger(project_name="BOEM", workspace="bw4sz")

#im = images_callback(n=20, every_n_epochs=25, savedir=os.path.join(savedir,"images"))

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

m.create_trainer(logger=comet_logger, accelerator="gpu", strategy="ddp", num_nodes=1, devices=devices)
m.trainer.fit(m)

# Save the model
m.trainer.save_checkpoint("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/checkpoints/{}.pl".format(comet_logger.experiment.id))

#results = m.evaluate(m.config["validation"]["csv_file"],m.config["validation"]["root_dir"])
#print(results)

# Gather the number of steps taken from all GPUs
global_steps = torch.tensor(m.trainer.global_step, dtype=torch.int32, device=m.device)
comet_logger.experiment.log_metric("global_steps", global_steps)

# Save profiler to comet
#comet_logger.experiment.log_asset(os.path.join(tmpdir,"profiler","profiler.txt"))
