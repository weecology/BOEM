import os
from typing import Dict, Iterable, List, Optional
import pandas as pd

from omegaconf import DictConfig

from src import label_studio as ls_mod
from src import sagemaker_gt as sm_mod


class BaseAnnotator:
    def upload(
        self,
        images: List[str],
        instance_name: str,
        preannotations: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        raise NotImplementedError

    def check_for_new_annotations(self, instance_name: str, image_dir: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    def gather_data(self, instance_name: str, image_dir: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError


class LabelStudioAnnotator(BaseAnnotator):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.sftp_client = ls_mod.create_sftp_client(**self.cfg.server)
        # Prepare per-flight directories for LS
        flight_name = os.path.basename(self.cfg.image_dir)
        self.cfg.annotation.label_studio.instances.train.csv_dir = os.path.join(self.cfg.annotation.label_studio.instances.train.csv_dir, flight_name)
        self.cfg.annotation.label_studio.instances.validation.csv_dir = os.path.join(self.cfg.annotation.label_studio.instances.validation.csv_dir, flight_name)
        self.cfg.annotation.label_studio.instances.review.csv_dir = os.path.join(self.cfg.annotation.label_studio.instances.review.csv_dir, flight_name)
        os.makedirs(self.cfg.annotation.label_studio.instances.train.csv_dir, exist_ok=True)
        os.makedirs(self.cfg.annotation.label_studio.instances.validation.csv_dir, exist_ok=True)
        os.makedirs(self.cfg.annotation.label_studio.instances.review.csv_dir, exist_ok=True)

    def upload(
        self,
        images: List[str],
        instance_name: str,
        preannotations: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        project_name = self.cfg.annotation.label_studio.instances[instance_name].project_name
        ls_mod.upload_to_label_studio(
            images=images,
            sftp_client=self.sftp_client,
            url=self.cfg.annotation.label_studio.url,
            project_name=project_name,
            images_to_annotate_dir=self.cfg.image_dir,
            folder_name=self.cfg.annotation.label_studio.folder_name,
            preannotations=preannotations,
        )

    def check_for_new_annotations(self, instance_name: str, image_dir: str) -> Optional[pd.DataFrame]:
        instance_cfg = self.cfg.annotation.label_studio.instances[instance_name]
        return ls_mod.check_for_new_annotations(
            url=self.cfg.annotation.label_studio.url,
            csv_dir=os.path.dirname(self.cfg.annotation.label_studio.instances.train.csv_dir),
            project_name=instance_cfg.project_name,
            image_dir=image_dir,
        )

    def gather_data(self, instance_name: str, image_dir: str) -> Optional[pd.DataFrame]:
        instance_csv = self.cfg.annotation.label_studio.instances[instance_name].csv_dir
        return ls_mod.gather_data(instance_csv, image_dir=image_dir)


class SageMakerAnnotator(BaseAnnotator):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def upload(
        self,
        images: List[str],
        instance_name: str,
        preannotations: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        # Build S3 URIs
        s3_prefix = getattr(self.cfg.annotation.sagemaker, "s3_prefix", "").rstrip("/")
        if not s3_prefix:
            raise ValueError("config.sagemaker.s3_prefix must be set for SageMaker uploads")

        basenames = [os.path.basename(p) for p in images]
        s3_uris = [os.path.join(s3_prefix, b) for b in basenames]

        out_dir = getattr(self.cfg.annotation.sagemaker, "output_dir", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        jobs_per_day = int(getattr(self.cfg.annotation.sagemaker, "jobs_per_day", 200))
        job_name = str(getattr(self.cfg.annotation.sagemaker, "job_name", "annotation"))

        roster_path = sm_mod.write_daily_roster(s3_uris=s3_uris, output_dir=out_dir)
        jobs_path, selected_uris = sm_mod.assign_jobs_from_roster(
            roster_path=roster_path, output_dir=out_dir, num_jobs=jobs_per_day
        )
        sm_mod.write_daily_metadata(selected_uris, output_dir=out_dir)
        sm_mod.write_daily_annotation_manifest(selected_uris, output_dir=out_dir, job_name=job_name)

        # Optional Globus upload
        try:
            dest_dir = str(getattr(self.cfg.annotation.sagemaker.globus, "dest_dir", "/daily"))
            client_id = getattr(self.cfg.annotation.sagemaker.globus, "native_app_client_id", None)
            source_collection_id = getattr(self.cfg.annotation.sagemaker.globus, "source_collection_id", None)
            dest_collection_id = getattr(self.cfg.annotation.sagemaker.globus, "dest_collection_id", None)
            sm_mod.globus_upload_files(
                local_paths=[roster_path, jobs_path, os.path.join(out_dir, f"{sm_mod._today_stamp()}_metadata.txt"), os.path.join(out_dir, f"{sm_mod._today_stamp()}_annotation_manifest.jsonl")],
                dest_collection_id=dest_collection_id,
                dest_dir=dest_dir,
                source_collection_id=source_collection_id,
                client_id=client_id,
            )
        except Exception:
            pass

    def check_for_new_annotations(self, instance_name: str, image_dir: str) -> Optional[pd.DataFrame]:
        instance_dir = self.cfg.annotation.sagemaker.instances[instance_name].csv_dir
        return sm_mod.gather_data(annotation_dir=instance_dir, image_dir=image_dir)

    def gather_data(self, instance_name: str, image_dir: str) -> Optional[pd.DataFrame]:
        instance_dir = self.cfg.annotation.sagemaker.instances[instance_name].csv_dir
        return sm_mod.gather_data(annotation_dir=instance_dir, image_dir=image_dir)


def get_annotator(cfg: DictConfig) -> BaseAnnotator:
    tool = getattr(cfg.annotation, "annotation_tool", "label_studio")
    if tool == "label_studio":
        return LabelStudioAnnotator(cfg)
    if tool == "sagemaker":
        return SageMakerAnnotator(cfg)
    raise ValueError(f"Unknown annotation tool: {tool}")


