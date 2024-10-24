# A python script to recursively read in a directory of images and rename them into a single directory with folder names added to file namesimport os
import shutil
import os

def rename_images(directory, rename_directory):
    """Flat a recursive directory of images into a single directory with folder names added to file names."""
    # Create a new directory to store the renamed images
    os.makedirs(rename_directory, exist_ok=True)

    # Recursively iterate through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(root, file)

            # Get the folder name
            folder_name = os.path.basename(root)

            # Create a new file name by adding the folder name as a prefix
            new_file_name = f"{folder_name}_{file}"

            # Move the file to the new directory with the new file name
            shutil.move(file_path, os.path.join(rename_directory, new_file_name))

    print("Images renamed successfully!")

if __name__ == "__main__":
    # Call the function with the directory path
    rename_images("/blue/ewhite/everglades/label_studio/airplane/images_to_annotate", "/blue/ewhite/everglades/label_studio/airplane/images_to_annotate/")
