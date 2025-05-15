import tqdm
import zipfile

def extract_dataset(zip_file_path: str, output_folder: str) -> None:
    """
    Extracts the dataset from the zip file to the output folder

    Args:
        zip_file_path: path to the zip file
        output_folder: path to the output folder

    Returns:
        None
    """

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        with tqdm.tqdm(total=total_files, desc='Extracting files') as pbar:
            for file in file_list:
                zip_ref.extract(file, output_folder)
                pbar.update(1)
    
    # zip_ref.extractall(output_folder)

if __name__ == '__main__':
    zip_file_path = 'data/raw/seabed_dataset.zip'
    output_folder = 'data/processed/'

    extract_dataset(zip_file_path, output_folder)


