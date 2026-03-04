from deepforest.model import CropModel
from pytorch_lightning import Trainer
from torchvision.datasets import ImageFolder
import numpy as np
import torch

import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix


cropmodel = CropModel.load_from_checkpoint("/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/checkpoints/3caaa23614c041eaa7edcc1231cf216b.ckpt")
root_dir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/crops/val"

# Create dataset
val_ds = ImageFolder(root=root_dir, transform=cropmodel.get_transform(augment=False))

# Create dataloader
crop_dataloader = cropmodel.predict_dataloader(val_ds)

# Run prediction
trainer = Trainer(devices=1, accelerator="gpu", max_epochs=1, logger=False, enable_checkpointing=False)
crop_results = trainer.predict(cropmodel, crop_dataloader)

label, score = cropmodel.postprocess_predictions(crop_results)

# Determine column names
label_column = "cropmodel_label"
score_column = "cropmodel_score"

label_name = [cropmodel.numeric_to_label_dict[x] for x in label]

# Calculate recall for "Tursiops truncatus"
true_label_idx = val_ds.class_to_idx["Tursiops truncatus"]
y_true = np.array([y for _, y in val_ds.samples])
y_pred = np.array(label)

# Recall: TP / (TP + FN)
tp = np.sum((y_true == true_label_idx) & (y_pred == true_label_idx))
fn = np.sum((y_true == true_label_idx) & (y_pred != true_label_idx))
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

print(f"Tursiops truncatus recall: {recall:.3f}")


# Method 1: Using torchmetrics
metric = MulticlassConfusionMatrix(num_classes=61)
metric.update(preds=torch.tensor(y_pred), target=torch.tensor(y_true))
fig, ax = metric.plot()
plt.title("Confusion Matrix")
plt.show()