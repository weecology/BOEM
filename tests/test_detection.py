import pandas as pd
from src.detection import preprocess_and_train

def test_preprocess_and_train(config):
    validation_df = pd.read_csv(config.detection_model.validation_csv_path)
    trained_model = preprocess_and_train(config, validation_df=validation_df)

    assert trained_model.trainer is not None
