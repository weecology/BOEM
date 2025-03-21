import cProfile
import pstats
import time
from src.detection import predict
import hydra
from deepforest import main
import glob
import torch
from PIL import Image
import numpy as np
from dask.distributed import Client, LocalCluster
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import CometLogger
from tabulate import tabulate

class JpgDataset_parallel(Dataset):
    def __init__(self, image_paths, patch_size, patch_overlap):
        """
        Args:
            images (list): List of loaded images as numpy arrays.
            patch_size (int): Size of the patches to extract.
            patch_overlap (int): Overlap between patches.
        """
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

    def _create_patches(self, image):
        image_tensor = torch.tensor(image).permute(2, 0, 1)  # Convert to (C, H, W)
        patches = image_tensor.unfold(1, self.patch_size, self.patch_size - self.patch_overlap).unfold(2, self.patch_size, self.patch_size - self.patch_overlap)
        patches = patches.contiguous().view(patches.shape[1] * patches.shape[2], 3, self.patch_size, self.patch_size)
        
        return patches

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = np.array(image)
        image = image / 255.0
        image = image.astype(np.float32)
        patches = self._create_patches(image)
        
        return patches

class JpgDataset(Dataset):
    def __init__(self, images, patch_size, patch_overlap):
        """
        Args:
            images (list): List of loaded images as numpy arrays.
            patch_size (int): Size of the patches to extract.
            patch_overlap (int): Overlap between patches.
        """
        self.images = images
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patches = self._create_patches()

    def _create_patches(self):
        all_patches = []
        for image in self.images:
            image_tensor = torch.tensor(image).permute(2, 0, 1)  # Convert to (C, H, W)
            patches = image_tensor.unfold(1, self.patch_size, self.patch_size - self.patch_overlap).unfold(2, self.patch_size, self.patch_size - self.patch_overlap)
            patches = patches.contiguous().view(patches.shape[1] * patches.shape[2], 3, self.patch_size, self.patch_size)
            all_patches.append(patches)
        return torch.cat(all_patches, dim=0)

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        patch = self.patches[idx]
        return patch

def load_images_in_parallel(image_paths, dask_client):
    def _load_jpg(jpg_path):
        image = Image.open(jpg_path)
        image = np.array(image)
        image = image / 255.0
        image = image.astype(np.float32)
        return image

    futures = [dask_client.submit(_load_jpg, jpg_path) for jpg_path in image_paths]
    images = dask_client.gather(futures)
    return images

def create_dataloader(ds, batch_size, num_workers=0, collate_fn=None, **kwargs):
    """
    Create a DataLoader for the JpgDataset.
    
    Args:
        ds (Dataset): Dataset to load.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
    
    Returns:
        DataLoader: DataLoader for the JpgDataset.
    """
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    
    return dataloader

def load_images_with_dask(m, image_paths, dask_client, batch_size=48):
    # Load images in parallel using Dask
    images = load_images_in_parallel(image_paths, dask_client)

    # Create a DataLoader with the loaded images
    ds = JpgDataset(images=images, patch_size=1000, patch_overlap=0)
    dl = create_dataloader(ds, batch_size=48, num_workers=0)
    results = m.trainer.predict(m, dl)

def load_images_serially(m, image_paths, batch_size=48, workers=0):
    m.config["workers"] = workers
    results = predict(m=m, image_paths=image_paths, batch_size=batch_size, patch_size=1000, patch_overlap=0)

def load_images_with_parallel_patches(m, image_paths, batch_size=2, workers=10):
    m.config["workers"] = workers
    ds = JpgDataset_parallel(image_paths=image_paths, patch_size=1000, patch_overlap=0)

    def collate_fn(batch):
        return torch.cat(batch, dim=0)

    dl = create_dataloader(ds, batch_size=batch_size, num_workers=workers, collate_fn=collate_fn)
    results = m.trainer.predict(m, dl)

@hydra.main(config_path="/home/b.weinstein/BOEM/conf", config_name="config")
def profile_predict(config):

    comet_logger = CometLogger(project_name=config.comet.project, workspace=config.comet.workspace)
    
    # Profile the function
    profiler = cProfile.Profile()
    profiler.enable()

    m = main.deepforest.load_from_checkpoint(config.detection_model.checkpoint)
    m.create_trainer(profiler="simple")

    # Get csv files from crop image dir
    images_paths = glob.glob(config.label_studio.images_to_annotate_dir + "/*")[:100]
    
    # Create a Dask client
    #dask_client = Client(LocalCluster(n_workers=5, processes=False, memory_limit='20GB'))

    timings = []

    # Load images with Dask
    #start_time = time.time()
    #load_images_with_dask(m, images_paths, dask_client)
    #end_time = time.time()
    #timings.append(["Load Images with Dask", end_time - start_time])

    # Load images serially
    start_time = time.time()
    load_images_serially(m, images_paths)
    end_time = time.time()
    timings.append(["Load Images Serially", end_time - start_time])

    # Load images with parallel patches
    start_time = time.time()
    load_images_with_parallel_patches(m=m, image_paths=images_paths)
    end_time = time.time()
    timings.append(["Load Images with Parallel Patches", end_time - start_time])

    profiler.disable()
    
    # Print the profiling results
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)
    
    # Print the time taken
    print(tabulate(timings, headers=["Option", "Time (seconds)"]))

    # Save .prof file
    profiler.dump_stats("profile_predict.prof")
    
if __name__ == "__main__":
    profile_predict()