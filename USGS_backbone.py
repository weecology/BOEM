from deepforest import main
import pandas as pd
import os
import tempfile
import comet_ml
from pytorch_lightning.loggers import CometLogger

df = pd.read_csv("/blue/ewhite/b.weinstein/BOEM/UBFAI Annotations/20231118/20231116_cropped_annotations.csv")
df.wat_label.value_counts()
df = df[df.wat_label.isin(["Bird","Cartilaginous Fish","Bony Fish","Mammal","Reptile"])]

# Combine Fish classes
df.loc[df.wat_label.isin(["Cartilaginous Fish","Bony Fish"]),"wat_label"] = "Fish"

# Construct padded crop name
df["image_path"] = df["bname_parent"] +"_" + df["tile_xtl"].astype(str) + "_" + df["tile_ytl"].astype(str) + "_" + df["tile_xbr"].astype(str) + "_" + df["tile_ybr"].astype(str) + ".JPG"

# Check if all images exist 
df["image_exists"] = df["image_path"].apply(lambda x: os.path.exists(os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Annotations/20231118/padded",x)))

df["xmin"] = df["xtl"]
df["ymin"] = df["ytl"]
df["xmax"] = df["xbr"]
df["ymax"] = df["ybr"]
df["label"] = df["wat_label"]

# Randomly split 80 - 20 for each class
train = df.groupby("wat_label").sample(frac=0.85)
test = df.drop(train.index)

# Write to tmp data directory
tmpdir = tempfile.mkdtemp()
train.to_csv(os.path.join(tmpdir,"train.csv"),index=False)
test.to_csv(os.path.join(tmpdir,"test.csv"),index=False)

# Initialize new Deepforest model ( the model that you will train ) with your classes
m = main.deepforest(config_args={"num_classes":4}, label_dict={"Bird":0,"Fish":1,"Mammal":2,"Reptile":3})

# Inatialize Deepforest model ( the model that you will modify its regression head ) 
deepforest_release_model = main.deepforest()
deepforest_release_model.load_model("weecology/deepforest-bird") # or load_model('weecology/deepforest-bird')

# Extract single class backbone that will have useful features for multi-class classification
m.model.backbone.load_state_dict(deepforest_release_model.model.backbone.state_dict())

# load regression head in the new model
m.model.head.regression_head.load_state_dict(deepforest_release_model.model.head.regression_head.state_dict())

m.config["train"]["csv_file"] = os.path.join(tmpdir,"train.csv")
m.config["train"]["root_dir"] = "/blue/ewhite/b.weinstein/BOEM/UBFAI Annotations/20231118/padded"
m.config["train"]["fast_dev_run"] = False
m.config["validation"]["csv_file"] = os.path.join(tmpdir,"test.csv")
m.config["validation"]["root_dir"] = "/blue/ewhite/b.weinstein/BOEM/UBFAI Annotations/20231118/padded"
m.config["batch_size"] = 6
m.config["train"]["epochs"] = 25
m.config["validation"]["val_accuracy_interval"] = 5
m.config["train"]["scheduler"]["params"]["eps"]  = 0
comet_logger = CometLogger(project_name="BOEM", workspace="bw4sz")

m.create_trainer(logger=comet_logger)
m.trainer.fit(m)

# Save the model
m.trainer.save_checkpoint("/blue/ewhite/b.weinstein/BOEM/UBFAI Annotations/checkpoints/{}.pl".format(comet_logger.experiment.id))
