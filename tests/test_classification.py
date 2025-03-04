import pandas as pd
from src.classification import preprocess_and_train_classification

def test_preprocess_and_train_classification(config, comet_logger):
    trained_model = preprocess_and_train_classification(
        config,
        comet_logger=comet_logger)

    assert trained_model.trainer is not None
