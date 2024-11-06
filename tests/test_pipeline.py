from src.pipeline import Pipeline

def test_pipeline_run(config):
    """Test complete pipeline run"""
    pipeline = Pipeline(cfg=config)
    pipeline.run()