import sys
from pathlib import Path
import tarfile
import gdown
import pandas as pd

config_dir = "../config"
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)

from config import google_folder_id, google_folder_id2

def download_gdrive_tar(file_id: str, tar_path: str | Path) -> Path:
    tar_path = Path(tar_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"

    if tar_path.exists() and not tarfile.is_tarfile(tar_path):
        print(f"Corrupted TAR detected at {tar_path}, removing and re-downloading...")
        tar_path.unlink()

    if not tar_path.exists():
        gdown.download(url, str(tar_path), quiet=False)

    return tar_path

def extract_tar(tar_path: str | Path, extract_dir: str | Path) -> Path:
    tar_path = Path(tar_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not tarfile.is_tarfile(tar_path):
        raise ValueError(f"File is not a valid TAR archive: {tar_path}")

    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=extract_dir)

    return extract_dir

def load_csvs_from_folder(
    folder: str | Path,
    pattern: str = "*.csv",
    **read_csv_kwargs
):
    folder = Path(folder)
    csv_files = sorted(folder.rglob(pattern))

    out = {}
    for fp in csv_files:
        out[str(fp)] = pd.read_csv(fp, **read_csv_kwargs)

    return out

def load_data(
    file_id: str = google_folder_id,
    work_dir: str | Path = "gdrive_tar_data",
    tar_name: str = "data.tar",
    pattern: str = "*.csv",
    **read_csv_kwargs
):
    work_dir = Path(work_dir)
    tar_path = work_dir / tar_name
    extracted_base = work_dir / "extracted"
    
    # Determine dataset type based on file_id
    if file_id == google_folder_id:
        dataset_type = "ARAG"
    elif file_id == google_folder_id2:
        dataset_type = "CAST"
    else:
        dataset_type = None
    
    # Check if data is already extracted (look for folder containing dataset type)
    data_exists = False
    data_folder = None
    
    if extracted_base.exists():
        for folder in extracted_base.iterdir():
            if folder.is_dir() and dataset_type and dataset_type in folder.name:
                data_exists = True
                data_folder = folder
                break
    
    if data_exists:
        print(f"Extracted data found at: {data_folder}")
    else:
        # Data not extracted, check if tar exists
        if tar_path.exists():
            print(f"Tar file found, extracting...")
            extract_tar(tar_path, extracted_base)
        else:
            # Tar doesn't exist either, download and extract
            print(f"Data not found, downloading...")
            tar_path = download_gdrive_tar(file_id, tar_path)
            print(f"Extracting data...")
            extract_tar(tar_path, extracted_base)
        
        # Find the extracted folder
        if extracted_base.exists():
            for folder in extracted_base.iterdir():
                if folder.is_dir() and dataset_type and dataset_type in folder.name:
                    data_folder = folder
                    break
    
    if data_folder is None or not data_folder.exists():
        raise FileNotFoundError(f"Data folder containing '{dataset_type}' not found in: {extracted_base}")
    
    print(f"Loading {dataset_type} data from: {data_folder}")
    data = load_csvs_from_folder(data_folder, pattern=pattern, **read_csv_kwargs)
    return data, data_folder