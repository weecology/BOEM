import os
from huggingface_hub import HfApi

class ModelDeployment:
    def __init__(self):
        """Initialize model deployment"""
        self.api = HfApi()
        self.repo_id = "weecology/boem-aerial-wildlife-detector"

    def deploy_model(self, model_path):
        """Deploy a model to the weecology Hugging Face space
        
        Args:
            model_path (str): Path to the model file to deploy
            
        Raises:
            FileNotFoundError: If model file does not exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Upload model to Hugging Face
        self.api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=self.repo_id,
            repo_type="model"
        )
        
        print(f"Model deployed to {self.repo_id}")
