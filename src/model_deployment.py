import os
from src.monitoring import Monitoring
from src.pipeline_evaluation import PipelineEvaluation
from huggingface_hub import HfApi, HfFolder

class ModelDeployment:
    def __init__(self):
        self.monitoring = Monitoring()
        self.pipeline_evaluation = PipelineEvaluation()
        self.hf_api = HfApi()
        self.hf_token = HfFolder.get_token()

    def upload_to_huggingface(self, model_path, repo_id):
        """
        Upload the successful checkpoint to Hugging Face.

        Args:
            model_path (str): The path to the model checkpoint.
            repo_id (str): The repository ID on Hugging Face.

        Returns:
            None
        """
        self.hf_api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=repo_id,
            token=self.hf_token
        )

