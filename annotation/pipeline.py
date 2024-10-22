import pandas as pd
from src import model, upload, active_learning
from deepforest.utilities import read_file
import os
from datetime import datetime
import json
import random
import tempfile

def config_pipeline(config, dask_client=None):
    iterate(dask_client=dask_client, **config)

def iterate(
        checkpoint_dir,
        images_to_annotate_dir,
        annotated_images_dir,
        test_csv,
        user,
        host,
        folder_name,
        key_filename,
        patch_size,
        patch_overlap,
        label_studio_url,
        label_studio_project_name,
        train_csv_folder,
        strategy="random",
        n_images=5,
        min_score=0.3,
        model_checkpoint=None,
        annotation_csv=None,
        force_run=False,
        skip_train=False,
        dask_client=None,
        target_labels=None,
        under_sample_ratio=0.5,
        comet_workspace=None,
        comet_project=None,
        labels=None,
        pool_limit=1000):
    """A Deepforest pipeline for rapid annotation and model iteration.

    Args:
        checkpoint_dir: The path to a directory for saving model checkpoints.
        images_to_annotate_dir: The path to a directory of images to annotate.
        annotated_images_dir: The path to a directory of annotated images.
        test_csv: The path to a CSV file containing annotations. Images are assumed to be in the same directory.
        user (str): The username for uploading images to the annotation platform.
        host (str): The host URL of the annotation platform.
        folder_name (str): The name of the folder to upload images to.
        model_checkpoint (str, optional): The path to the model checkpoint file. Defaults to None.
        annotation_csv (str, optional): The path to the CSV file containing annotations. Defaults to None. This will skip checking the server for debugging
        patch_size: The size of the image patches to predict on for main.deepforest.predict_tile.
        patch_overlap: The amount of overlap between image patches.
        label_studio_url: The URL of the Label Studio server.
        label_studio_project_name: The name of the Label Studio project.
        train_csv_folder: The path to a directory of CSV files containing annotations.
        target_labels: A list of target labels to filter images by. Defaults to None.
        under_sample_ratio: The ratio of images to remove from the top two classes. Defaults to 0.1.
        min_score: The minimum score for a prediction to be included in the annotation platform.
        force_run: If True, will run the pipeline even if there are no new annotations. Defaults to False.
        skip_train: If True, will skip training the model. Defaults to False.
        strategy: The strategy for choosing images. Available strategies are:
            - "random": Choose images randomly from the pool.
            - "most-detections": Choose images with the most detections based on predictions.
            - "target-labels": Choose images with specified labels.
        n_images: The number of images to choose.
        dask_client: A dask distributed client for parallel prediction. Defaults to None.
        labels: A list of labels to filter by. Defaults to None.
        pool_limit: The maximum number of images to consider. Defaults to 1000.
        comet_workspace: The comet workspace for logging. Defaults to None.
        comet_project: The comet project name for logging. Defaults to None.
    Returns:
        None
    """
    # Check event for there new annotations
    # Download labeled annotations
    sftp_client = upload.create_client(user=user, host=host, key_filename=key_filename)
    label_studio_project = upload.connect_to_label_studio(url=label_studio_url, project_name=label_studio_project_name)
    new_annotations = upload.download_completed_tasks(label_studio_project=label_studio_project, train_csv_folder=train_csv_folder)
   
   # Move annotated images out of local pool
    if new_annotations is not None:
        upload.move_images(src_dir=images_to_annotate_dir, dst_dir=annotated_images_dir, annotations=new_annotations)
        # Get any images from the server that are not in the images_to_annotate_dir
        for image in new_annotations["image_path"].unique():
            if not os.path.exists(os.path.join(annotated_images_dir, image)):
                upload.download_images(sftp_client=sftp_client, image_names=[image], folder_name=folder_name, local_image_dir=annotated_images_dir)
        
        upload.delete_completed_tasks(label_studio_project=label_studio_project)

    # Choose new images to annotate
    label_studio_annotations = upload.gather_data(train_csv_folder, labels=labels)
 
    if force_run:
        if annotation_csv:
            label_studio_annotations = pd.read_csv(annotation_csv)

        # Load existing model
        if model_checkpoint:
            m = model.load(model_checkpoint, annotations=label_studio_annotations)
        elif os.path.exists(checkpoint_dir):
            m = model.get_latest_checkpoint(checkpoint_dir, label_studio_annotations)
        else:
            evaluation = None
            
        # Train model and save checkpoint
        jsons = label_studio_project.export_tasks()
        
        # For each json, save and name based on id key
        for record in jsons:
            id = record["id"]
            with open("{}/label_studio/label_studio_export_{}.json".format(train_csv_folder, id), "w") as file:
                # dump dict as json
                json_str = json.dumps(record)
                file.write(json_str)

        if not skip_train:
            # HOT FIX, if label is nan, set as first label, it will be ignored in training and set to 0.
            label_studio_annotations["label"] = label_studio_annotations["label"].fillna(label_studio_annotations["label"].unique()[0])
            # Fill in missing values with 0 in xmin, xmax, ymin, ymax
            label_studio_annotations[["xmin", "xmax", "ymin", "ymax"]] = label_studio_annotations[["xmin", "xmax", "ymin", "ymax"]].fillna(0)

            if not test_csv:
                train_df, validation_df = model.create_train_test(label_studio_annotations, under_sample_ratio=under_sample_ratio)

            else:
                train_df = label_studio_annotations
            
            # Chunk images
            tmpdir = tempfile.gettempdir()
            crop_dir = os.path.join(images_to_annotate_dir, "crops")
            
            crop_annotations_train = model.preprocess_images(
                train_df, 
                save_dir=crop_dir, 
                root_dir=annotated_images_dir, 
                patch_size=patch_size, 
                patch_overlap=patch_overlap
            )
            crop_annotations_train.to_csv(os.path.join(tmpdir, "train.csv"), index=False)

            crop_annotations_test = model.preprocess_images(
                validation_df,  
                save_dir=crop_dir, 
                root_dir=annotated_images_dir, 
                patch_size=patch_size, 
                patch_overlap=patch_overlap
            )
            crop_annotations_test.to_csv(os.path.join(tmpdir, "test.csv"), index=False)
            test_csv = os.path.join(tmpdir, "test.csv")

            m.config["validation"]["csv_file"] = test_csv
            m.config["validation"]["root_dir"] = crop_dir
            before_evaluation = model.evaluate(m, test_csv=test_csv, image_root_dir=crop_dir)
            deepforest_eval_before = m.evaluate(test_csv, root_dir=crop_dir, iou_threshold=0.4)
            print(before_evaluation)
            print(deepforest_eval_before)
            
            m = model.train(
                model=m,
                train_annotations=crop_annotations_train,
                test_annotations=crop_annotations_test,
                train_image_dir=crop_dir,
                comet_project=comet_project,
                comet_workspace=comet_workspace
         )

            # Choose new images to annotate
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            if test_csv:
                # Create a directory to save prediction plots in checkpoint dir
                prediction_dir = os.path.join(checkpoint_dir,timestamp)
                os.makedirs(prediction_dir, exist_ok=True)
                evaluation = model.evaluate(m, test_csv=test_csv, image_root_dir=crop_dir)

                print(evaluation)

            # Save a checkpoint using timestamp
            model_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.ckpt")
            m.trainer.save_checkpoint(model_checkpoint)
        else:
            model_checkpoint = None

        # Choose local images to annotate
        images = active_learning.choose_images(
            image_dir=images_to_annotate_dir,
            evaluation=None,
            strategy=strategy,
            n=n_images,
            m=m,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            min_score=min_score,
            dask_client=dask_client,
            model_path=model_checkpoint,
            target_labels=target_labels,
            pool_limit=pool_limit
        )

        if len(images) == 0:
             raise ValueError("No new images selected to annotate")
        
        # Predict images if annotations don't already exist
        if os.path.exists(images_to_annotate_dir + "/annotations.csv"):
            all_preannotations = read_file(pd.read_csv(images_to_annotate_dir + "/annotations.csv"))
            # Create a list of predictions for each image, only include images that are in the pool
            preannotations = []
            for image in images:
                image_annotations = all_preannotations[all_preannotations["image_path"] == image]
                preannotations.append(image_annotations)
        else:
            preannotations = model.predict(
                m=m,
                model_path=model_checkpoint,
                image_paths=images,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                min_score=min_score,
                dask_client=None
            )
            
        # Upload images to annotation platform
        upload.upload_images(sftp_client=sftp_client, images=images, folder_name=folder_name)
        upload.import_image_tasks(label_studio_project=label_studio_project, image_names=images, local_image_dir=images_to_annotate_dir, predictions=preannotations)

    else:
        print("No new annotations")
