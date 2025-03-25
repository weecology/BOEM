# Prepare USGS backbone
import pandas as pd
import os
import glob
from deepforest.preprocess import split_raster
from deepforest.utilities import read_file
import torch
import argparse
import random
import numpy as np
from src.cluster import start
from dask.distributed import as_completed
import shutil
from src.label_studio import gather_data

# Parse arguments
parser = argparse.ArgumentParser(description="Train DeepForest model")
parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
parser.add_argument("--workers", type=int, default=0, help="Number of workers for data loading")
args = parser.parse_args()

# Use parsed arguments
batch_size = args.batch_size
workers = args.workers

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

df = pd.read_csv("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/20250203_total.csv")
df.label.value_counts()

# Construct padded crop name
df["image_path"] = df["bname_parent"] + ".JPG"

# Check if all images exist remove any that do not exist
df["image_exists"] = df["image_path"].apply(lambda x: os.path.exists(os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/images_parent",x)))
df = df[df["image_exists"]]

df["xmin"] = df["left"]
df["ymin"] = df["top"]
df["xmax"] = df["left"] + df["width"]
df["ymax"] = df["top"] + df["height"]

os.makedirs("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops", exist_ok=True)
crop_annotations =[]
regenerate_crops = False
if regenerate_crops:
    client = start(cpus=5, mem_size="40GB")
    futures = []

    def process_image(image_annotations):
        x = image_annotations.image_path.unique()[0]
        filename = os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops", x.replace(".JPG", ".csv"))
        if os.path.exists(filename):
            return pd.read_csv(filename)
        try:
            split_raster(
                annotations_file=image_annotations,
                patch_size=1000,
                patch_overlap=0,
                path_to_raster=os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/images_parent", x),
                root_dir="/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/images_parent",
                base_dir="/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops",
                allow_empty=False)
            return filename
        except Exception as e:
            print(f"Error processing {x}: {e}")
            return None

    for x in df.image_path.unique():
        image_annotations = df[df["image_path"] == x]
        futures.append(client.submit(process_image, image_annotations))

    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            crop_annotations.append(result)

crop_annotations = glob.glob("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops/*.csv")
crop_annotations = [pd.read_csv(x) for x in crop_annotations]

crop_annotations = pd.concat(crop_annotations)

# Background classes as negatives
crop_annotations.loc[crop_annotations['label'].isin(["Algae", "Boat", "Buoy"]), ['xmin', 'xmax', 'ymin', 'ymax', 'label']] = [0, 0, 0, 0, "Object"]

# All other as "Object"
crop_annotations.loc[~crop_annotations['label'].isin(["Algae", "Boat", "Buoy"]), 'label'] = "FalsePositive"

# Drop duplicates for False Positives only
falsepositives = crop_annotations[crop_annotations['label'] == "FalsePositive"]
falsepositives = falsepositives.drop_duplicates(subset=['xmin', 'xmax', 'ymin', 'ymax'])

# Drop any falsepositive images that occur in the same image as a true positive by image_path
true_positives = crop_annotations[crop_annotations['label'] != "FalsePositive"]
falsepositives = falsepositives[~falsepositives['image_path'].isin(true_positives['image_path'])]
crop_annotations = pd.concat([crop_annotations[crop_annotations['label'] != "FalsePositive"], falsepositives])
crop_annotations["label"] = "Object"

images = crop_annotations.image_path.unique()
random.shuffle(images)
train_images = images[:int(len(images)*0.90)]
test_images = images[int(len(images)*0.90):]
train = crop_annotations[crop_annotations["image_path"].isin(train_images)]
test = crop_annotations[crop_annotations["image_path"].isin(test_images)]

# Now sweep any existing flight crops to be added on.
# Recursively search for .csv files and copy them along with their images
# for csv_file in glob.glob("/blue/ewhite/b.weinstein/BOEM/detection/crops/**/*.csv", recursive=True):
#     # Read the csv file
#     annotations = pd.read_csv(csv_file)
    
#     # Copy the csv file to the destination directory
#     destination_csv = os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops", os.path.basename(csv_file))
#     annotations.to_csv(destination_csv, index=False)
    
#     # Copy the associated images
#     for image_path in annotations["image_path"].unique():
#         source_image = os.path.join("/blue/ewhite/b.weinstein/BOEM/detection/crops", image_path)
#         destination_image = os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops", os.path.basename(image_path))
#         if os.path.exists(source_image):
#             os.makedirs(os.path.dirname(destination_image), exist_ok=True)
#             # check if the image already exists
#             if not os.path.exists(destination_image):
#                 shutil.copy2(source_image, destination_image)
    
#     # Add the annotations to crop_annotations
#     crop_annotations = pd.concat([crop_annotations, annotations])

# Make sure any train stay in train

train_csvs = glob.glob("/blue/ewhite/b.weinstein/BOEM/annotations/train/*/*.csv", recursive=True)
val_csvs = glob.glob("/blue/ewhite/b.weinstein/BOEM/annotations/validation/*/*.csv", recursive=True)

flight_train = pd.concat([pd.read_csv(x) for x in train_csvs])
flight_val = pd.concat([pd.read_csv(x) for x in val_csvs])

# Write directory
savedir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"

train = read_file(train.drop(columns="geometry"))
test = read_file(test.drop(columns="geometry"))

flight_train = read_file(flight_train, savedir)
flight_val = read_file(flight_val, savedir)

train = pd.concat([train, flight_train])
test = pd.concat([test, flight_val])

# Check if all images exist remove any that do not exist
train["image_exists"] = train["image_path"].apply(lambda x: os.path.exists(os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops",x)))
train = train[train["image_exists"]]

# Check if all images exist remove any that do not exist
test["image_exists"] = test["image_path"].apply(lambda x: os.path.exists(os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops",x)))
test = test[test["image_exists"]]

test["image_exists"].value_counts()
train["image_exists"].value_counts()

train.to_csv(os.path.join(savedir,"train.csv"),index=False)
test.to_csv(os.path.join(savedir,"test.csv"),index=False)
