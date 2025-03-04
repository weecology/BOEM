import pandas as pd
from src.detection import preprocess_and_train

def test_preprocess_and_train(config, comet_logger):
    trained_model = preprocess_and_train(config, comet_logger=comet_logger)
    assert trained_model.trainer is not None
