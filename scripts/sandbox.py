from deepforest import main
import shutil
import os
from deepforest import get_data

def test_predict_tile_batch_uses_global_image_indices(tmp_path):
    """Batch strategy must assign image_path using global dataset indices, not batch position.
    """
    m = main.deepforest()
    source = get_data("OSBS_029.png")
    num_images = 5
    paths = []
    for i in range(num_images):
        dest = tmp_path / f"image_{i}.png"
        shutil.copy(source, dest)
        paths.append(str(dest))
    m.config.train.fast_dev_run = False
    m.create_trainer()
    m.load_model("weecology/deepforest-tree")
    prediction = m.predict_tile(
        path=paths,
        patch_size=300,
        patch_overlap=0,
        dataloader_strategy="batch",
    )
    unique_paths = prediction.image_path.unique().tolist()
    assert len(unique_paths) == num_images
    expected_basenames = sorted(os.path.basename(p) for p in paths)
    assert sorted(unique_paths) == expected_basenames