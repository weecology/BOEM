# BOEM Active Learning Pipeline

A machine learning pipeline for detecting and annotating birds in aerial imagery.

## Project Structure

```
BOEM/
├── boem_conf/                         # Hydra configuration
│   ├── boem_config.yaml               # Top-level defaults
│   ├── classification_model/
│   │   ├── finetune.yaml              # DeepForest classifier settings
│   │   ├── hierarchical.yaml          # H-CAST classifier settings
│   │   └── USGS.yaml                  # Legacy classification config
│   ├── server/
│   │   └── serenity.yaml              # Label Studio server creds
│   ├── Detection/                     # (reserved)
│   ├── Images/                        # (reserved)
│   └── with/                          # (reserved)
│
├── src/                               # Pipeline source
│   ├── active_learning.py             # Pool prediction, selection, human review
│   ├── classification.py              # DeepForest crop classification train/ft
│   ├── detection.py                   # DeepForest detection train/infer
│   ├── hierarchical.py                # H-CAST wrapper + inference adapter
│   ├── label_studio.py                # LS auth, SFTP upload, task import/export
│   ├── pipeline.py                    # Orchestrates detection + classification
│   ├── pipeline_evaluation.py         # Eval metrics and summaries
│   ├── data_ingestion.py              # Data loading utilities
│   ├── data_processing.py             # Crop writing and preprocessing
│   ├── visualization.py               # Crop visualization, video helpers
│   ├── sagemaker_gt.py                # Optional SageMaker Ground Truth pipeline
│   ├── propagate.py                   # Result propagation utilities
│   ├── cluster.py                     # Clustering helpers
│   └── utils.py                       # Shared helpers
│
├── tamu_hcast/                        # H-CAST module (third-party)
│   ├── cast_models/                   # CAST model definitions
│   ├── deit/                          # Training/eval scripts and datasets
│   ├── data/                          # Dataset metadata
│   └── README.md
│
├── tests/                             # Pytest suite
│   ├── test_active_learning.py
│   ├── test_classification.py
│   ├── test_data_ingestion.py
│   ├── test_data_processing.py
│   ├── test_detection.py
│   ├── test_pipeline.py
│   ├── test_pipeline_evaluation.py
│   ├── test_propagate.py
│   └── test_visualization.py
│
├── main.py                            # Entry point (Hydra)
├── pyproject.toml                     # uv project metadata and deps
├── environment.yml                    # Legacy conda env (deprecated)
├── README.md                          # This file
└── scripts, utilities, and notebooks (various)
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

## Installation (uv)

1. Clone the repository:
```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

2. Install Python deps with uv (PyPI packages only):
```bash
uv venv -p 3.10
uv pip install -e .
```

3. Install PyTorch (CUDA 12.1) and torchvision from NVIDIA wheels:
```bash
uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

4. Install DGL matching your CUDA/PyTorch:
```bash
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
```

Notes:
- If using CPU-only, install CPU wheels instead of cu121. See PyTorch and DGL docs.
- H-CAST requires `timm==0.4.12` and OpenCV, already included in `pyproject.toml`.

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

### Configuration with Hydra

Hydra configs live under `boem_conf/`. The top-level defaults are in `boem_conf/boem_config.yaml`.

- Switch classification backend:
  - DeepForest (default):
    - `classification_model.backend: deepforest`
  - Hierarchical (H-CAST):
    - `classification_model.backend: hierarchical`
    - `classification_model.checkpoint: /path/to/best_checkpoint.pth` (optional; auto-discovers under `tamu_hcast/`)

Override at runtime:
```bash
python main.py classification_model=hierarchical
```

### Label Studio Integration

The pipeline can automatically upload images, import preannotations, and fetch completed annotations from Label Studio.

Prereqs:
- Get your Label Studio API key and save it to `.label_studio.config` at repo root in the format:
  ```
  api_key=YOUR_API_KEY
  ```
- Configure Label Studio in Hydra (`boem_conf/boem_config.yaml`):
  - `label_studio.url`: server URL (e.g., `https://labelstudio.naturecast.org/`)
  - `label_studio.folder_name`: remote folder path (server-side) where images are uploaded via SFTP
  - `label_studio.instances.{train,validation,review}.project_name`: Label Studio project titles to target
  - `label_studio.instances.{train,validation,review}.csv_dir`: where downloaded annotations CSVs are written
  - SFTP credentials (`server` group) for uploads: `user`, `host`, `key_filename`

Run:
```bash
python main.py
```

What happens:
- The pipeline connects to Label Studio using `LABEL_STUDIO_API_KEY` (set automatically from `.label_studio.config`).
- New predictions are uploaded via SFTP to `folder_name/input`, then imported as tasks with bounding box + taxonomy preannotations.
- Completed tasks are downloaded to per-flight CSVs under the configured `csv_dir` folders, then removed from Label Studio to avoid duplication.
- Images marked as completed are archived server-side.

Manual overrides:
- To disable checking Label Studio, set `check_annotations: false` in `boem_conf/boem_config.yaml`.

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