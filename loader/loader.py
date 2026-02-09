from pathlib import Path
import tarfile
import gdown
import pandas as pd
from key import id
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
    file_id: str = id,
    work_dir: str | Path = "gdrive_tar_data",
    tar_name: str = "data.tar",
    pattern: str = "*.csv",
    **read_csv_kwargs
):
    work_dir = Path(work_dir)
    tar_path = download_gdrive_tar(file_id, work_dir / tar_name)
    extracted = extract_tar(tar_path, work_dir / "extracted")
    data = load_csvs_from_folder(extracted, pattern=pattern, **read_csv_kwargs)
    return data, extracted