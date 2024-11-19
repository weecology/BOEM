import pytest
import pandas as pd
from deepforest import main
from deepforest.model import CropModel
from src.model import preprocess_and_train

@pytest.mark.parametrize("model_type", ["detection", "classification"])
def test_preprocess_and_train(config, model_type, crop_model):
    if model_type == "detection":
        model = main.deepforest()
    else:
        model = CropModel()
    model.create_trainer(fast_dev_run=True)

    validation_df = pd.read_csv(config.detection_model.validation_csv_path)
    trained_model = preprocess_and_train(config, m=model, validation_df=validation_df, model_type=model_type)
