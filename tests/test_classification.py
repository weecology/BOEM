import pandas as pd
from src.classification import preprocess_and_train_classification

def test_preprocess_and_train_classification(config):
    validation_df = pd.read_csv(config.detection_model.validation_csv_path)

    trained_model = preprocess_and_train_classification(config, validation_df)

    assert trained_model.trainer is not None
