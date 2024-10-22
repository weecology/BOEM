# ML Workflow Manager

ML Workflow Manager is a high-level Python package for managing machine learning workflows. It provides a modular structure for data ingestion, processing, model training, evaluation, deployment, and monitoring. It also includes an annotation module based on the AirborneFieldGuide project.

## Installation

You can install the ML Workflow Manager using pip:


## Usage

To run the main workflow and the annotation pipeline:

```bash
python main.py
```

## Running Tests

To run the tests for the ML Workflow Manager:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests using pytest:
   ```bash
   pytest tests/
   ```

This will run all the tests in the `tests/` directory and display the results.

## Annotation Module

The `annotation` module is based on the AirborneFieldGuide project. It provides additional functionality for annotating airborne data. To use this module, you can import it in your Python scripts:

```python
from annotation.pipeline import config_pipeline

# Use the config_pipeline function to run the annotation workflow
config_pipeline(your_config)
```

For more details on how to use the annotation module, please refer to the AirborneFieldGuide documentation.
