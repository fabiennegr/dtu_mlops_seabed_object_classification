import os
import shutil
from tests import _PATH_DATA
from dtu_mlops_team94.data.make_dataset import extract_dataset  # Test the make_dataset.py script
import tempfile

# module_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../dtu_mlops_team94")
# sys.path.append(module_path)

DATASET_ZIP_PATH = os.path.join(_PATH_DATA, "raw/seabed_dataset.zip")
PROCESSED_DATA_PATH = os.path.join(_PATH_DATA, "processed/")


# Verify that the data folder exists
def test_data_folder_exists():
    assert os.path.exists(_PATH_DATA), "The data folder does not exist"


# Verify that the compressed dataset exists
def test_seabed_dataset_compressed_exists():
    assert os.path.exists(DATASET_ZIP_PATH), "The data/raw/seabed_dataset.zip does not exist, please do 'dvc pull'"


# Verify that the compressed dataset is not empty
def test_seabed_dataset_compressed_not_empty():
    assert os.path.getsize(DATASET_ZIP_PATH) > 0, "The data/raw/seabed_dataset.zip is empty, please do 'dvc pull'"


def test_make_dataset():
    temp_folder = tempfile.mkdtemp(dir=PROCESSED_DATA_PATH)
    extract_dataset(DATASET_ZIP_PATH, temp_folder)
    assert os.path.exists(temp_folder), "The temporary folder has not been created"

    # Test the extract_dataset() function
    extract_dataset(DATASET_ZIP_PATH, temp_folder)

    # Verify that the temporary folder is not empty
    assert os.path.getsize(temp_folder) > 0, "The temporary output_folder is empty (no files extracted)"

    # Verify that there is a folder called 'seabed_dataset' inside the temporary folder
    assert os.path.exists(
        os.path.join(temp_folder, "seabed_dataset")
    ), "The temporary output_folder does not contain the 'seabed_dataset' folder"

    # Verify that there is a folder called 'imgs' inside the 'seabed_dataset' folder
    assert os.path.exists(
        os.path.join(temp_folder, "seabed_dataset/imgs")
    ), "The temporary output_folder does not contain the 'seabed_dataset/imgs' folder"

    # Verify that there is at least one image inside the 'seabed_dataset/imgs' folder
    assert (
        len(os.listdir(os.path.join(temp_folder, "seabed_dataset/imgs"))) > 0
    ), "The temporary output_folder does not contain any images"

    # Verify that there is a folder called 'labels' inside the 'seabed_dataset' folder
    assert os.path.exists(
        os.path.join(temp_folder, "seabed_dataset/labels")
    ), "The temporary output_folder does not contain the 'seabed_dataset/labels' folder"

    # Verify that there is a file called 'labels.csv' inside the 'seabed_dataset/labels' folder
    assert os.path.exists(
        os.path.join(temp_folder, "seabed_dataset/labels/labels.csv")
    ), "The temporary output_folder does not contain the 'seabed_dataset/labels/labels.csv' file"

    # Delete the temporary folder and its contents
    shutil.rmtree(temp_folder)
