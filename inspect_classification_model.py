
from deepforest.dataset import BoundingBoxDataset
import os
from torch.functional import F
import pandas as pd

raster_path = "/blue/ewhite/b.weinstein/BOEM/sample_flight/JPG_2024_Jan27/C4_L2_F2072_T20240127_135614_149.jpg" 
results = flightline_predictions[flightline_predictions['image_path'] == "C4_L2_F2072_T20240127_135614_149.jpg"]

bounding_box_dataset = BoundingBoxDataset(
        results,
        root_dir=os.path.dirname(raster_path),
        transform=None,
        augment=False)

image = bounding_box_dataset[0].unsqueeze(0)

trained_classification_model.eval()
outputs = trained_classification_model.model(image)
F.sigmoid(outputs) == trained_classification_model(image)


trained_classification_model.model.eval()
predicted_class = []
predicted_prob = []
for batch in trained_classification_model.val_dataloader():
    x, y = batch
    outputs = trained_classification_model.model(x)
    yhat = F.softmax(outputs, dim=1)
    for i in range(len(yhat)):
        predicted_class.append(yhat[i].argmax().item())
        predicted_prob.append(yhat[i].max().item())
        
predicted_frame = pd.DataFrame({"predicted_class":predicted_class, "predicted_prob":predicted_prob})


trained_classification_model.create_trainer()
trained_classification_model.load_from_disk(train_dir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/crops/train",
                                            val_dir = "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/classification/crops/val",)
trained_classification_model.trainer.validate(trained_classification_model, trained_classification_model.val_dataloader()) # validate the model