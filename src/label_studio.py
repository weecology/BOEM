import json
import paramiko
import os
import datetime
import pandas as pd
from label_studio_sdk import Client
import glob
import shutil
from PIL import Image
from deepforest.utilities import read_file

from src.active_learning import format_ensemble_suggestion_line


def get_taxonomy_leaf_paths(taxonomy_path):
    """Load transformed_taxonomy.json and return a dict mapping leaf alias -> full path (list of aliases from root to leaf).

    Label Studio Taxonomy with apiUrl expects the prediction value to be the full path so the dropdown shows the correct selection.
    """
    with open(taxonomy_path, encoding="utf-8") as f:
        data = json.load(f)
    result = {}

    def visit(node, path_so_far):
        alias = node.get("alias") or node.get("value")
        if not alias:
            return
        path = path_so_far + [alias]
        if node.get("isLeaf", not node.get("children")) or not node.get("children"):
            result[alias] = path
        for c in node.get("children", []):
            visit(c, path)

    for item in data.get("items", []):
        visit(item, [])

    return result

def upload_to_label_studio(
    images,
    sftp_client,
    url,
    project_name,
    images_to_annotate_dir,
    folder_name,
    preannotations,
    species_to_genus=None,
    import_chunk_size=100,
):
    """
    Upload images to Label Studio and import image tasks.

    Args:
        images (list): List of image paths to upload, full paths
        url (str): The URL of the Label Studio server.
        sftp_client (paramiko.SFTPClient): The SFTP client for uploading images.
        project_name (str): The name of the Label Studio project.
        images_to_annotate_dir (str): The path to the directory of images to annotate.
        folder_name (str): The name of the folder to upload images to.
        preannotations (list): List of preannotations for the images.

    Returns:
        None
    """
    default_label_config = (
        '<View>\n'
        '  <Image name="image" value="$image"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        '    <Label value="Object" background="#D4380D"/>\n'
        '    <Label value="FalsePositive" background="#FFA39E"/>\n'
        '  </RectangleLabels>\n'
        '  <Text name="text" value="Select Taxonomy for Classification"/>\n'
        '  <Taxonomy name="taxonomy" perRegion="true" minWidth="600px" toName="image" '
        'apiUrl="https://raw.githubusercontent.com/weecology/BOEM/refs/heads/main/transformed_taxonomy.json" />\n'
        '</View>'
    )
    label_studio_project = connect_to_label_studio(url=url, project_name=project_name, label_config=default_label_config)
    upload_images(sftp_client=sftp_client, images=images, folder_name=folder_name)
    import_image_tasks(
        label_studio_project=label_studio_project,
        image_names=images,
        local_image_dir=images_to_annotate_dir,
        predictions=preannotations,
        species_to_genus=species_to_genus,
        chunk_size=import_chunk_size,
    )

def check_for_new_annotations(url, project_name, csv_dir, image_dir):
    """
    Check for new annotations from Label Studio, move annotated images, and gather new images to annotate.

    Args:
        url (str): The URL of the Label Studio server.
        project_name (str): The name of the Label Studio project.
        csv_dir (str): The path to the folder containing CSV files.
        image_dir (str): The path to the folder containing images.

    Returns:
        DataFrame: A DataFrame containing the gathered annotations.
    """
    label_studio_project = connect_to_label_studio(url=url, project_name=project_name)
    new_annotations = download_completed_tasks(label_studio_project=label_studio_project, csv_dir=csv_dir)
   
   # Move annotated images out of local pool
    if new_annotations is not None:
        try:
            delete_completed_tasks(label_studio_project=label_studio_project)
        except Exception as e:
            print(f"Error deleting completed tasks: {e}")
    
    else:
        print("No new annotations")
        return None

    # Choose new images to annotate
    label_studio_annotations = gather_data(csv_dir, image_dir)

    #clean_inactive_images(sftp_client, label_studio_project, folder_name)

    return label_studio_annotations
 

def label_studio_bbox_format(local_image_dir, preannotations, taxonomy_path=None):
    """Create prediction payload for a single image for the Label Studio API.

    When taxonomy_path is set, taxonomy values use the full path from root to leaf (from
    transformed_taxonomy.json) so the Taxonomy dropdown shows the species label, not "Object".
    """
    predictions = []
    original_width = Image.open(os.path.join(local_image_dir, os.path.basename(preannotations.image_path.unique()[0]))).size[0]
    original_height = Image.open(os.path.join(local_image_dir, os.path.basename(preannotations.image_path.unique()[0]))).size[1]

    taxonomy_paths = get_taxonomy_leaf_paths(taxonomy_path) if taxonomy_path and os.path.exists(taxonomy_path) else None

    for index, row in preannotations.iterrows():
        region_id = "region" + str(index)
        box_result = {
            "value": {
                "x": row['xmin']/original_width*100,
                "y": row['ymin']/original_height*100,
                "width": (row['xmax'] - row['xmin'])/original_width*100,
                "height": (row['ymax'] - row['ymin'])/original_height*100,
                "rotation": 0,
                "rectanglelabels": [row["label"]]
            },
            "id": region_id,
            "source": "$image",
            "model_version": row["comet_id"],
            "score": row["score"],
            "to_name": "image",
            "type": "rectanglelabels",
            "from_name": "label",
            "original_width": original_width,
            "original_height": original_height
        }
        predictions.append(box_result)

        # Taxonomy value must be full path from root to leaf so the dropdown shows the species, not "Object"
        species = row.get("cropmodel_label", row.get("label", "Object"))
        if taxonomy_paths and species in taxonomy_paths:
            path_list = taxonomy_paths[species]
        else:
            path_list = [species]
        class_results = {
            "value": {
                "taxonomy": [path_list]
            },
            "id": region_id,
            "source": "$image",
            "to_name": "image",
            "type": "taxonomy",
            "from_name": "taxonomy",
            "original_width": original_width,
            "original_height": original_height
        }
        predictions.append(class_results)

    return {"result": predictions}


def format_prediction_summary_for_task(
    prediction_df: pd.DataFrame,
    species_to_genus: dict | None = None,
) -> str:
    """Format crop + H-CAST predictions for Label Studio task data (prediction_summary).

    When species_to_genus is provided and hcast_* columns exist, appends ensemble suggestion line
    (species agreement, genus fallback on mismatch, or ambiguous note).
    """
    if prediction_df is None or prediction_df.empty:
        return "No detections for this image."
    lines = []
    for i, (_, row) in enumerate(prediction_df.iterrows(), 1):
        crop_label = row.get("cropmodel_label", row.get("label", "\u2014"))
        score = row.get("score", row.get("cropmodel_score", ""))
        if isinstance(score, (int, float)):
            parts = [f"Crop {i}: {crop_label} (score={score:.2f})"]
        else:
            parts = [f"Crop {i}: {crop_label}"]
        if "hcast_species" in row and pd.notna(row.get("hcast_species")):
            sp = row["hcast_species"]
            gn = row.get("hcast_genus")
            fa = row.get("hcast_family")
            sp_s = row.get("hcast_species_score")
            parts.append(f"  H-CAST: species={sp}")
            if pd.notna(gn):
                parts.append(f", genus={gn}")
            if pd.notna(fa):
                parts.append(f", family={fa}")
            if sp_s is not None and isinstance(sp_s, (int, float)):
                parts.append(f" (species_score={sp_s:.2f})")
        lines.append("".join(parts))
        ens = format_ensemble_suggestion_line(row, species_to_genus)
        if ens:
            lines.append(f"  {ens}")
    return "\n".join(lines) if lines else "No detections for this image."


# check_if_complete label studio images are done
def check_if_complete(annotations):
    """Check if any new images have been labeled.
    
    Returns:
        bool: True if new images have been labeled, False otherwise.
    """

    if annotations.shape[0] > 0:
        return True
    else:
        return False

# con for a json that looks like 
#{'id': 539, 'created_username': ' vonsteiny@gmail.com, 10', 'created_ago': '0\xa0minutes', 'task': {'id': 1962, 'data': {...}, 'meta': {}, 'created_at': '2023-01-18T20:58:48.250374Z', 'updated_at': '2023-01-18T20:58:48.250387Z', 'is_labeled': True, 'overlap': 1, 'inner_id': 381, 'total_annotations': 1, ...}, 'completed_by': {'id': 10, 'first_name': '', 'last_name': '', 'email': 'vonsteiny@gmail.com'}, 'result': [], 'was_cancelled': False, 'ground_truth': False, 'created_at': '2023-01-30T21:43:35.447447Z', 'updated_at': '2023-01-30T21:43:35.447460Z', 'lead_time': 29.346, 'parent_prediction': None, 'parent_annotation': None}
    
def convert_json_to_dataframe(x):
    # Loop through annotations and convert to pandas {'original_width': 6016, 'original_height': 4008, 'image_rotation': 0, 'value': {'x': 94.96474718276704, 'y': 22.132321974413898, 'width': 1.7739074476466308, 'height': 2.2484415320942235, 'rotation': 0, 'rectanglelabels': [...]}, 'id': 'UeovfQERjL', 'from_name': 'label', 'to_name': 'image', 'type': 'rectanglelabels', 'origin': 'manual'}
    results = []
    for annotation in x:
        xmin = annotation["value"]["x"]/100 * annotation["original_width"]
        ymin = annotation["value"]["y"]/100 * annotation["original_height"]
        xmax = (annotation["value"]["width"]/100 + annotation["value"]["x"]/100 ) * annotation["original_width"]
        ymax = (annotation["value"]["height"]/100 + annotation["value"]["y"]/100) * annotation["original_height"]
        
        if "taxonomy" in annotation["value"]:
            label = annotation["value"]["taxonomy"][0][-1]
        else:
            label = annotation["value"]["rectanglelabels"][0]

        # Create dictionary
        result = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "label": label,
        }

        # Append to list
        results.append(result)

    df = pd.DataFrame(results)

    # Drop the 'Object' labels for the taxonomy row
    df = df[~(df.label == "Object")]
        
    return df
        

# Move images from images_to_annotation to images_annotated 
def move_images(annotations, src_dir, dst_dir):
    """Move images from the images_to_annotate folder to the images_annotated folder.
    Args:
        annotations (list): A list of annotations.
    
    Returns:
        None
    """
    images = annotations.image_path.unique()
    for image in images:
        src = os.path.join(src_dir, os.path.basename(image))
        dst = os.path.join(dst_dir, os.path.basename(image))
                           
        try:
            shutil.move(src, dst)
        except FileNotFoundError:
            continue

def gather_data(annotation_dir, image_dir):
    """Gather data from a directory of CSV files.
    Args:
        annotation_dir (str): The directory containing the CSV files.
        image_dir: Location of images on disk
    
    Returns:
        pd.DataFrame: A DataFrame containing the data.
    """ 
    csvs = glob.glob(os.path.join(annotation_dir,"*.csv"))
    df = []
    for x in csvs:
        image_annotations = pd.read_csv(x)
        image_annotations["filename"] = x
        df.append(image_annotations )
    
    if len(df) == 0:
        return None
    df = pd.concat(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # For images with multipe filenames, take the most recent one
    most_recent_annotation = df.groupby("image_path")["filename"].max().reset_index()
    df_most_recent = df.merge(most_recent_annotation, on=["image_path","filename"])
    df_most_recent = read_file(df_most_recent)

    return df_most_recent

def get_api_key():
    """Get Label Studio API key from config file"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               '.label_studio.config')
    if not os.path.exists(config_path):
        return None

    with open(config_path, 'r') as f:
        for line in f:
            if line.startswith('api_key'):
                return line.split('=')[1].strip()
    return None

def connect_to_label_studio(url, project_name, label_config=None):
    """Connect to the Label Studio server.
    Args:
        project_name (str): The name of the project to connect to.
        label_config (str, optional): The label configuration for the project. Defaults to None.
    Returns:
        str: The URL of the Label Studio server.
    """
    ls = Client(url=url, api_key=os.environ["LABEL_STUDIO_API_KEY"])
    ls.check_connection()

    # Look up existing name
    projects = ls.list_projects()
    project = [x for x in projects if x.get_params()["title"] == project_name]

    if len(project) == 0:
        # Create a project with the specified title and labeling configuration
        project = ls.create_project(
            title=project_name,
            label_config=label_config
        )
    else:
        project = project[0]

    return project

def create_project(ls, project_name):
    ls.create_project(title=project_name)

def create_sftp_client(user, host, key_filename):
    # Download annotations from Label Studio
    # SSH connection with a user prompt for password
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, key_filename=key_filename)
    sftp = ssh.open_sftp()

    return sftp

def delete_completed_tasks(label_studio_project):
    # Delete completed tasks. Must use the same listing as download_completed_tasks,
    # or the pipeline deletes tasks it never downloaded.
    for task in get_annotated_tasks(label_studio_project):
        label_studio_project.delete_task(task["id"])

def import_image_tasks(
    label_studio_project,
    image_names,
    local_image_dir,
    predictions=None,
    species_to_genus=None,
    chunk_size=100,
):
    """
    Import image tasks into Label Studio project.

    Args:
        label_studio_project (LabelStudioProject): The Label Studio project to import tasks into.
        image_names (list): List of image names to import as tasks.
        local_image_dir (str): The local directory where the images are stored.
        predictions (dict, optional): Dictionary of predictions with image basename as keys. Defaults to None.

    Returns:
        None
    """
    tasks = []
    for image_name in image_names:
        print(f"Importing {image_name} into Label Studio")
        basename = os.path.basename(image_name)
        flight_name = os.path.dirname(image_name).split("/")[-1]

        # Compute relative path from local_image_dir to image_name to preserve directory structure
        # This handles datasets with nested subdirectories (e.g., NEAQ) correctly
        try:
            relative_path = os.path.relpath(image_name, local_image_dir)
        except ValueError:
            # If relpath fails (different drives on Windows, etc), fall back to basename
            relative_path = basename

        data_dict = {
            'image': os.path.join("/data/local-files/?d=BOEM/input/", basename),
            'flight_name': flight_name,
            'image_relative_path': relative_path,
        }
        if predictions is not None:
            prediction = predictions.get(basename, pd.DataFrame())
            data_dict["prediction_summary"] = format_prediction_summary_for_task(
                prediction,
                species_to_genus=species_to_genus,
            )
            # Skip predictions if there are none
            if prediction.empty:
                result_dict = []
            else:
                result_dict = [label_studio_bbox_format(local_image_dir, prediction)]
            upload_dict = {"data": data_dict, "predictions": result_dict}
        else:
            data_dict["prediction_summary"] = "No prediction data for this task."
            upload_dict = {"data": data_dict}
        tasks.append(upload_dict)

    for i in range(0, len(tasks), chunk_size):
        label_studio_project.import_tasks(tasks[i:i + chunk_size])

def get_annotated_tasks(label_studio_project):
    """Every task carrying an annotation.

    get_labeled_tasks() is a filtered view that under-reports — it has been observed
    returning 218 of 248 annotated tasks, silently leaving the rest behind. get_tasks()
    is the authoritative listing, so filter that instead.
    """
    return [t for t in label_studio_project.get_tasks() if t.get("annotations")]


def download_completed_tasks(label_studio_project, csv_dir):
    labeled_tasks = get_annotated_tasks(label_studio_project)
    if not labeled_tasks:
        print("No new annotations")
        return None
    else:
        images, labels = [], []
    for labeled_task in labeled_tasks:
        task_annotations = labeled_task.get('annotations') or []
        if not task_annotations:
            print(f"Skipping task {labeled_task.get('id')} with no annotations")
            continue

        # Use relative_path if available (preserves directory structure for nested datasets like NEAQ),
        # otherwise fall back to basename for backwards compatibility with old Label Studio exports
        task_data = labeled_task['data']
        if 'image_relative_path' in task_data:
            image_path = task_data['image_relative_path']
        else:
            image_path = os.path.basename(task_data['image'])

        images.append(image_path)
        label_json = task_annotations[0]["result"]
        if len(label_json) == 0:
            result = {
                    "image_path": image_path,
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0,
                    "label": 0,
                    "annotator":task_annotations[0]["created_username"],
                    'flight_name': labeled_task['data']['flight_name']
                }
            result = pd.DataFrame(result, index=[0])
        else:
            result = convert_json_to_dataframe(label_json)
            result["image_path"] = image_path
            result["annotator"] = task_annotations[0]["created_username"]
            result["flight_name"] = labeled_task['data']['flight_name']
        labels.append(result)

    if not labels:
        print("No new annotations")
        return None

    annotations =  pd.concat(labels)
    print("There are {} new annotations".format(annotations.shape[0]))

    # Save csv in dir with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for flight_name, group in annotations.groupby("flight_name"):
        flight_dir = os.path.join(csv_dir, flight_name)
        os.makedirs(flight_dir, exist_ok=True)
        flight_csv_path = os.path.join(flight_dir, f"{timestamp}.csv")
        group.to_csv(flight_csv_path, index=False)

    return annotations

def download_images(sftp_client, image_names, local_image_dir, folder_name):
    """
    Download images from a remote server using SFTP.

    Args:
        sftp_client (SFTPClient): An instance of the SFTPClient class representing the SFTP connection.
        image_names (list): A list of image file names to be downloaded.
        local_image_dir (str): The local directory where the images will be saved.
        folder_name (str): The name of the folder on the remote server where the images are located.

    Returns:
        None

    Raises:
        Any exceptions that may occur during the file transfer.

    """
    # SCP file transfer
    for image_name in image_names:
        remote_path = os.path.join(folder_name, "input", image_name)
        local_path = os.path.join(local_image_dir, image_name)
        try:
            sftp_client.get(remote_path, local_path)
        except FileNotFoundError:
            continue
        
        print(f"Downloaded {image_name} successfully")

def upload_images(sftp_client, images, folder_name):
    """
    Uploads a list of images to a remote server using SFTP.

    Args:
        sftp_client (SFTPClient): An instance of the SFTPClient class representing the SFTP connection.
        images (list): A list of image file paths to be uploaded.
        folder_name (str): The name of the folder on the remote server where the images will be uploaded.

    Returns:
        None

    Raises:
        Any exceptions that may occur during the file transfer.

    """
    # SCP file transfer with retry mechanism
    for image in images:
        retries = 3
        for attempt in range(retries):
            try:
                sftp_client.put(image, os.path.join(folder_name, "input", os.path.basename(image)))
                print(f"Uploaded {image} successfully")
                break
            except EOFError as e:
                if attempt < retries - 1:
                    print(f"Retrying upload for {image} due to EOFError (attempt {attempt + 1}/{retries})...")
                else:
                    print(f"Failed to upload {image} after {retries} attempts. Error: {e}")

def remove_annotated_images_remote_server(sftp_client, annotations, folder_name):
    """Remove images that have been annotated on the Label Studio server."""
    # Delete images using SSH
    for image in annotations.image_path.unique():
        remote_path = os.path.join(folder_name, "input", os.path.basename(image))
        # Archive annotations using SSH
        archive_annotation_path = os.path.join(folder_name, "archive", os.path.basename(image))
        # sftp check if dir exists
        try:
            sftp_client.listdir(os.path.join(folder_name, "archive"))
        except FileNotFoundError:
            raise FileNotFoundError("The archive directory {} does not exist.".format(os.path.join(folder_name, "archive")))
        
        sftp_client.rename(remote_path, archive_annotation_path)
        print(f"Archived {image} successfully")


def clean_inactive_images(sftp_client, label_studio_project, folder_name):
    """
    Check the names of active image tasks in Label Studio and delete all images
    from the input folder on the remote server that are not in the active tasks.

    Args:
        sftp_client (SFTPClient): An instance of the SFTPClient class representing the SFTP connection.
        label_studio_project (LabelStudioProject): The Label Studio project to check active tasks.
        folder_name (str): The name of the folder on the remote server where the images are stored.

    Returns:
        None
    """
    # Get active tasks from Label Studio
    active_tasks = label_studio_project.get_tasks()
    active_image_names = {os.path.basename(task['data']['image']) for task in active_tasks}

    # List images in the input folder on the remote server
    remote_input_folder = os.path.join(folder_name, "input")
    try:
        remote_images = sftp_client.listdir(remote_input_folder)
    except FileNotFoundError:
        print(f"The input folder {remote_input_folder} does not exist.")
        return

    # Identify and delete images not in active tasks
    for image in remote_images:
        if image not in active_image_names:
            remote_path = os.path.join(remote_input_folder, image)
            try:
                sftp_client.remove(remote_path)
                print(f"Deleted inactive image: {image}")
            except FileNotFoundError:
                print(f"Image {image} not found during deletion.")