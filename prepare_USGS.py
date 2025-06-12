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
import PIL.Image
from pytorch_lightning.loggers import CometLogger

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
unique_images = df["image_path"].unique()
images_that_exist = [os.path.exists(os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/images_parent", x)) for x in unique_images]

print("removing {} images that do not exist".format(len(unique_images) - sum(images_that_exist)))

unique_images = unique_images[images_that_exist]
df = df[df["image_path"].isin(unique_images)]

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
train_images = images[:int(len(images)*0.95)]
test_images = images[int(len(images)*0.95):]
train = crop_annotations[crop_annotations["image_path"].isin(train_images)]
test = crop_annotations[crop_annotations["image_path"].isin(test_images)]

#Now sweep any existing flight crops to be added on.
#Recursively search for .csv files and copy them along with their images
flight_annotations = [] 
for csv_file in glob.glob("/blue/ewhite/b.weinstein/BOEM/detection/crops/**/*.csv", recursive=True):
    
    # Read the csv file
    annotations = pd.read_csv(csv_file)
    
    # Copy the csv file to the destination directory
    destination_csv = os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops", os.path.basename(csv_file))
    if os.path.exists(destination_csv):
        print(f"Skipping {csv_file}, already exists in {destination_csv}")
        # Read the copied annotations   
        annotations = pd.read_csv(destination_csv)
    else:
        annotations.to_csv(destination_csv, index=False)

        # Copy the associated images (faster: batch check existence, skip already copied)
        image_paths = annotations["image_path"].unique()
        dest_dir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"
        pool = glob.glob("/blue/ewhite/b.weinstein/BOEM/detection/crops/**/*.png", recursive=True)
        
        for src in image_paths:
            # Find the source image path
            src_path = [x for x in pool if os.path.basename(x) == os.path.basename(src)][0]
            dst = os.path.join(dest_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)
    
    flight_annotations.append(annotations)

flight_annotations = pd.concat(flight_annotations)
# Make sure any train stay in train
train_csvs = glob.glob("/blue/ewhite/b.weinstein/BOEM/annotations/train/*/*.csv", recursive=True)
reviewed_csvs = glob.glob("/blue/ewhite/b.weinstein/BOEM/annotations/review/*/*.csv", recursive=True)
val_csvs = glob.glob("/blue/ewhite/b.weinstein/BOEM/annotations/validation/*/*.csv", recursive=True)

flight_train = pd.concat([pd.read_csv(x) for x in train_csvs + reviewed_csvs])
flight_val = pd.concat([pd.read_csv(x) for x in val_csvs])

# Write directory
savedir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops"

train = read_file(train.drop(columns="geometry"))
test = read_file(test.drop(columns="geometry"))

flight_train = read_file(flight_train, savedir)
flight_val = read_file(flight_val, savedir)

# get the detection crops 
train_parent_images = [os.path.splitext(x)[0] for x in flight_train["image_path"]]
test_parent_images = [os.path.splitext(x)[0] for x in flight_val["image_path"]]

flight_annotations["bname_parent"] = flight_annotations["image_path"].apply(lambda x: "_".join(x.split("_")[:-1]))
train_crop_annotations = flight_annotations[flight_annotations["bname_parent"].isin(train_parent_images)]
test_crop_annotations = flight_annotations[flight_annotations["bname_parent"].isin(test_parent_images)]

combined_train = pd.concat([train, train_crop_annotations])
combined_test = pd.concat([test, test_crop_annotations])

# Count the number of images
combined_train["empty_image"] = (combined_train["xmin"] == 0) & (combined_train["xmax"] == 0) & (combined_train["ymin"] == 0) & (combined_train["ymax"] == 0)
combined_test["empty_image"] = (combined_test["xmin"] == 0) & (combined_test["xmax"] == 0) & (combined_test["ymin"] == 0) & (combined_test["ymax"] == 0)

# Get the first 200 empty test images
#empty_test_images = combined_test[combined_test["empty_image"]].sample(n=200, random_state=seed)

# Remove empty test images for now
combined_test = combined_test[~combined_test["empty_image"]]

combined_train["label"] = "Object"
combined_test["label"] = "Object"

size_2000 = 0
images = []
for x in combined_test.image_path.unique():
    img = PIL.Image.open(os.path.join("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops",x))
    # count the size 2000 index
    if img.size[0] == 2000:
        size_2000 += 1
        images.append(x)

print("Number of size 2000 images in test set:", size_2000)

# remove the size 2000 images from the test set
combined_test = combined_test[~combined_test.image_path.isin(images)]

mini_train = combined_train.groupby("label").apply(lambda x: x.groupby("image_path").head(1)).reset_index(drop=True)
mini_train = mini_train.groupby("label").head(500).reset_index(drop=True)

mini_test = combined_test.groupby("label").apply(lambda x: x.groupby("image_path").head(1)).reset_index(drop=True)
mini_test = mini_test.groupby("label").head(50).reset_index(drop=True)

# save the mini sets
mini_train.to_csv(os.path.join(savedir, "mini_train.csv"), index=False)
mini_test.to_csv(os.path.join(savedir, "mini_test.csv"), index=False)

# Resave the train and test sets with the new label column
combined_train.to_csv(os.path.join(savedir,"train.csv"), index=False)
combined_test.to_csv(os.path.join(savedir,"test.csv"), index=False)