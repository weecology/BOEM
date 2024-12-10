import pandas as pd
from src.classification import preprocess_and_train_classification

def test_preprocess_and_train_classification(config):
    trained_model = preprocess_and_train_classification(config)

    assert trained_model.trainer is not None
