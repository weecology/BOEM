from logging import warn
from deepforest import main, preprocess
import os
from pytorch_lightning.loggers import CometLogger
import tempfile
import warnings
import glob
import pandas as pd
import dask.array as da
from deepforest import visualize
from deepforest.utilities import read_file
import random

def evaluate(model, test_csv, image_root_dir):
    """Evaluate a model on labeled images.
    
    Args:
        model (main.deepforest): A trained deepforest model.
        test_csv (str): The path to a CSV file containing annotations.
        image_root_dir (str): The directory containing the test images.
    Returns:
        dict: A dictionary of evaluation metrics.
    """
    # create trainer
    model.create_trainer()
    model.config["validation"]["csv_file"] = test_csv
    model.config["validation"]["root_dir"] = image_root_dir
    results = model.trainer.validate(model)

    return results

def load(path, annotations = None):
    """Load a trained model from disk.
    
    Args:
        path (str): The path to a model checkpoint.
    
    Returns:
        main.deepforest: A trained deepforest model.
    """
    # Check classes
    if path == "tree":
        snapshot = main.deepforest()
        snapshot.use_release()
    elif path == "bird":
        snapshot = main.deepforest(label_dict={"Bird":0})
        snapshot.use_bird_release()
    else:   
        snapshot = main.deepforest.load_from_checkpoint(path)

    if not annotations is None:
        num_labels = len(annotations.label.unique())
        if num_labels != len(snapshot.label_dict):
            snapshot = extract_backbone(snapshot, annotations)
    
    return snapshot
        
def extract_backbone(snapshot, annotations):
    warnings.warn("The number of classes in the model does not match the number of classes in the annotations. The backbone will be extracted and retrained.")
    new_labels = annotations.label.unique()
    new_labels = new_labels[~pd.isnull(new_labels)]

    label_dict = {value: index for index, value in enumerate(new_labels)}
    m = main.deepforest(num_classes=len(new_labels), label_dict=label_dict, config_file="deepforest_config.yml")
    m.model.backbone.load_state_dict(snapshot.model.backbone.state_dict())
    m.model.head.regression_head.load_state_dict(snapshot.model.head.regression_head.state_dict())

    return m

def process_image(image_path, annotation_df, root_dir, save_dir, limit_empty_frac, patch_size, patch_overlap):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    crop_csv = "{}.csv".format(os.path.join(save_dir, image_name))
    if os.path.exists(crop_csv):
        warn("Crops for {} already exist in {}. Skipping.".format(crop_csv, save_dir))
        return pd.read_csv(crop_csv)
    full_path = os.path.join(root_dir, image_path)
    crop_annotation = preprocess.split_raster(
        path_to_raster=full_path,
        annotations_file=annotation_df,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        save_dir=save_dir,
        root_dir=root_dir,
        allow_empty=True
    )

    return crop_annotation
    
def preprocess_images(annotations, root_dir, save_dir, limit_empty_frac=0.1, patch_size=450, patch_overlap=0):
    """Cut images into GPU friendly chunks"""
    crop_annotations = []

    for image_path in annotations.image_path.unique():
        crop_annotation = process_image(image_path, annotation_df=annotations, root_dir=root_dir, save_dir=save_dir, 
                                        limit_empty_frac=limit_empty_frac, patch_size=patch_size, patch_overlap=patch_overlap)
        crop_annotations.append(crop_annotation)

    crop_annotations = pd.concat(crop_annotations)
    crop_annotation_empty = crop_annotations.loc[crop_annotations.xmin==0]
    crop_annotation_non_empty = crop_annotations.loc[crop_annotations.xmin!=0]
    crop_annotation_empty = crop_annotation_empty.sample(frac=limit_empty_frac)
    crop_annotations = pd.concat([crop_annotation_empty, crop_annotation_non_empty])
    
    return crop_annotations
  
def create_train_test(annotations, train_test_split = 0.1, under_sample_ratio=0.4):
    tmpdir = tempfile.gettempdir()
    # split train images into 90% train and 10% validation for each class as much as possible
    test_images = []
    validation_df = None
    for label in annotations.label.value_counts().sort_values().index.values:
        # is the current count already higher than 10% of the total count?
        if validation_df is not None:
            if label in validation_df.label.value_counts().index:
                if validation_df.label.value_counts()[label] > annotations.label.value_counts()[label] * train_test_split:
                    continue
        images = list(annotations[annotations["label"] == label]["image_path"].unique())
        label_test_images = random.sample(images, int(len(images)*train_test_split))
        test_images.extend(label_test_images)
        validation_df = annotations[annotations["image_path"].isin(test_images)]

    validation_df.to_csv(os.path.join(tmpdir, f"random_validation_{label}.csv"), index=False)

    # Save validation csv
    train_df = annotations[~annotations["image_path"].isin(test_images)]
    
    ## Undersample top classes by selecting most diverse images

    # Find images that only have top two most common classes
    top_two_classes = annotations.label.value_counts().index[:2]
    top_two_labels = train_df[train_df["label"].isin(top_two_classes)]
    
    # remove images that have any other classes
    top_two_images= train_df[train_df.image_path.isin(top_two_labels.image_path.unique())]
    with_additional_species = top_two_images[~top_two_images["label"].isin(top_two_classes)].image_path.unique()
    images_to_remove = [x for x in top_two_images.image_path.unique() if x not in with_additional_species][:int(len(with_additional_species)*under_sample_ratio)]
    train_df = train_df[~train_df["image_path"].isin(images_to_remove)]
    
    return train_df, validation_df

def train(model, train_annotations, test_annotations, train_image_dir, comet_project=None, comet_workspace=None):
    """Train a model on labeled images.
    Args:
        image_paths (list): A list of image paths.
        train_annotations (pd.DataFrame): A DataFrame containing annotations.
        test_annotations (pd.DataFrame): A DataFrame containing annotations.
        train_image_dir (str): The directory containing the training images.
        comet_project (str): The comet project name for logging. Defaults to None.
    
    Returns:
        main.deepforest: A trained deepforest model.
    """
    tmpdir = tempfile.gettempdir()

    train_annotations.to_csv(os.path.join(tmpdir,"train.csv"), index=False)
    model.config["train"]["csv_file"] = os.path.join(tmpdir,"train.csv")
    model.config["train"]["root_dir"] = train_image_dir

    if comet_project:
        comet_logger = CometLogger(project_name=comet_project, workspace=comet_workspace)
        comet_logger.experiment.log_parameters(model.config)
        comet_logger.experiment.log_table("train.csv", train_annotations)
        comet_logger.experiment.log_table("test.csv", test_annotations)

        model.create_trainer(logger=comet_logger)
    else:
        model.create_trainer()
    
    with comet_logger.experiment.context_manager("train_images"):
        non_empty_train_annotations = train_annotations[train_annotations.xmin!=0]
        sample_train_annotations = non_empty_train_annotations[non_empty_train_annotations.image_path.isin(non_empty_train_annotations.image_path.sample(5))]
        sample_train_annotations = read_file(sample_train_annotations, root_dir=train_image_dir)
        sample_train_annotations["label"] = sample_train_annotations.apply(lambda x: model.label_dict[x["label"]], axis=1)
        filenames = visualize.plot_prediction_dataframe(sample_train_annotations, savedir=tmpdir, root_dir=train_image_dir)
        for filename in filenames:
            comet_logger.experiment.log_image(filename)

    with comet_logger.experiment.context_manager("test_images"):
        non_empty_test_annotations = test_annotations[test_annotations.xmin!=0]
        sample_test_annotations = non_empty_test_annotations[non_empty_test_annotations.image_path.isin(non_empty_test_annotations.image_path.sample(5))]
        sample_test_annotations = read_file(sample_test_annotations, root_dir=train_image_dir)
        sample_test_annotations["label"] = sample_test_annotations.apply(lambda x: model.label_dict[x["label"]], axis=1)
        filenames = visualize.plot_prediction_dataframe(sample_test_annotations, savedir=tmpdir, root_dir=train_image_dir)
        for filename in filenames:
            comet_logger.experiment.log_image(filename)

    with comet_logger.experiment.context_manager("PRE-training prediction"):
        for image_path in test_annotations.image_path.sample(5):
            prediction = model.predict_image(path = os.path.join(train_image_dir, image_path))
            print(prediction)
            if prediction is None:
                continue
            visualize.plot_results(prediction, savedir=tmpdir)
            comet_logger.experiment.log_image(os.path.join(tmpdir, image_path))

    model.trainer.fit(model)
    with comet_logger.experiment.context_manager("post-training prediction"):
        for image_path in test_annotations.image_path.sample(5):
            prediction = model.predict_image(path = os.path.join(train_image_dir, image_path))
            if prediction is None:
                continue
            visualize.plot_results(prediction, savedir=tmpdir)
            comet_logger.experiment.log_image(os.path.join(tmpdir, image_path))

    return model

def get_latest_checkpoint(checkpoint_dir, annotations):
    #Get model with latest checkpoint dir, if none exist make a new model
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir,"*.ckpt"))
        if len(checkpoints) > 0:
            checkpoints.sort()
            checkpoint = checkpoints[-1]
            m = load(checkpoint)
        else:
            warn("No checkpoints found in {}".format(checkpoint_dir))
            label_dict = {value: index for index, value in enumerate(annotations.label.unique())}
            m = main.deepforest(config_file="Airplane/deepforest_config.yml", label_dict=label_dict)
    else:
        os.makedirs(checkpoint_dir)
        m = main.deepforest(config_file="Airplane/deepforest_config.yml")
    
    return m

def _predict_list_(image_paths, min_score, patch_size, patch_overlap, model_path, m=None):
    if model_path:
        m = load(model_path)
    else:
        if m is None:
            raise ValueError("A model or model_path is required for prediction.")
    
    # if no trainer, create one
    if m.trainer is None:
        m.create_trainer()
    
    predictions = []
    for image_path in image_paths:
            try:
                prediction = m.predict_tile(raster_path=image_path, return_plot=False, patch_size=patch_size, patch_overlap=patch_overlap)
            except ValueError:
                continue
            prediction = prediction[prediction.score > min_score]
            predictions.append(prediction)
    
    return predictions

def predict(image_paths, patch_size, patch_overlap, min_score, m=None, model_path=None,dask_client=None):
    """Predict bounding boxes for images
    Args:
        m (main.deepforest): A trained deepforest model.
        image_paths (list): A list of image paths.  
    Returns:
        list: A list of image predictions.
    """

    if dask_client:
        # load model on each client
        def update_sys_path():
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dask_client.run(update_sys_path)

        # Load model on each client
        dask_pool = da.from_array(image_paths, chunks=len(image_paths)//len(dask_client.ncores()))
        blocks = dask_pool.to_delayed().ravel()
        block_futures = []
        for block in blocks:
            block_future = dask_client.submit(_predict_list_,
                                              image_paths=block.compute(),
                                              patch_size=patch_size,
                                              patch_overlap=patch_overlap,
                                              min_score=min_score,
                                              model_path=model_path,
                                              m=m)
            block_futures.append(block_future)
        # Get results
        predictions = []
        for block_result in block_futures:
            block_result = block_result.result()
            predictions.append(pd.concat(block_result))
    else:
        predictions = _predict_list_(image_paths=image_paths, patch_size=patch_size, patch_overlap=patch_overlap, min_score=min_score, model_path=model_path, m=m)

    return predictions