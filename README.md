# Bird Detection Pipeline

A machine learning pipeline for detecting and annotating birds in aerial imagery.

## Project Structure

```
project_root/
│
├── src/                      # Source code for the ML pipeline
│   ├── __init__.py
│   ├── data_ingestion.py    # Data loading and preparation
│   ├── data_processing.py   # Data preprocessing and transformations
│   ├── model_training.py    # Model training functionality
│   ├── pipeline_evaluation.py # Pipeline and model evaluation metrics
│   ├── model_deployment.py  # Model deployment utilities
│   ├── monitoring.py        # Monitoring and logging functionality
│   ├── reporting.py         # Report generation for pipeline results
│   ├── pre_annotation_prediction.py  # Pre-annotation model predictions
│   └── annotation/          # Annotation-related functionality
│       ├── __init__.py
│       └── pipeline.py      # Annotation pipeline implementation
│
├── tests/                   # Test files for each component
│   ├── test_data_ingestion.py
│   ├── test_data_processing.py
│   ├── test_model_training.py
│   ├── test_pipeline_evaluation.py
│   ├── test_model_deployment.py
│   ├── test_monitoring.py
│   ├── test_reporting.py
│
├── conf/                    # Configuration files
│   └── config.yaml         # Main configuration file
│
├── main.py                 # Main entry point for the pipeline
├── run_ml_workflow.sh      # Script to run pipeline in Serenity container
├── requirements.txt        # Project dependencies
├── .gitignore             # Git ignore file
├── CONTRIBUTING.md        # Contributing guidelines
├── LICENSE                # Project license
└── README.md             # This file
```

## Components

### Source Code (`src/`)

- **data_ingestion.py**: Handles data loading and initial preparation
- **data_processing.py**: Implements data preprocessing and transformations
- **model_training.py**: Contains model training logic
- **pipeline_evaluation.py**: Evaluates pipeline performance and model metrics
- **model_deployment.py**: Manages model deployment
- **monitoring.py**: Provides monitoring and logging capabilities
- **reporting.py**: Generates reports for pipeline results
- **annotation/**: Contains annotation-related functionality
  - **pipeline.py**: Implements the annotation pipeline

### Tests (`tests/`)

Contains test files corresponding to each component in `src/`. Uses pytest for testing.

### Configuration (`conf/`)

Contains YAML configuration files managed by Hydra:
- **config.yaml**: Main configuration file defining pipeline parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline

Using the Serenity container:
```bash
./run_ml_workflow.sh your-branch-name
```

Or directly with Python:
```bash
python main.py
```

### Running Tests

```bash
pytest tests/
```

## Configuration

The pipeline uses Hydra for configuration management. Main configuration options are defined in `conf/config.yaml`.

Example configuration:
```yaml
data:
  input_dir: "path/to/input"
  output_dir: "path/to/output"

model:
  type: "classification"
  parameters:
    learning_rate: 0.001
    batch_size: 32

pipeline:
  steps:
    - data_ingestion
    - data_processing
    - model_training
    - evaluation
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Dependencies

Key dependencies include:
- Hydra
- PyTorch
- NumPy
- Pandas
- Pytest

See `requirements.txt` for a complete list.

## Development

### Code Organization

- Each component is a separate module in the `src/` directory
- Tests mirror the source code structure in the `tests/` directory
- Configuration is managed through Hydra
- Monitoring and logging are integrated throughout the pipeline using comet

### Testing

- Tests are written using pytest
- Each component has its own test file
- Run tests with `pytest tests/`

### Adding New Components

1. Create a new module in `src/`
2. Add corresponding test file in `tests/`
3. Update configuration in `conf/config.yaml`
4. Update `main.py` to integrate the new component
5. Create a branch and push your changes to the remote repository
6. Create a pull request to merge your changes into the main branch