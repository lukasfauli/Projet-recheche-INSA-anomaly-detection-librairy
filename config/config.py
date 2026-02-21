from dotenv import load_dotenv
import os
import sys

load_dotenv('../.env')
google_folder_id = os.getenv('google_data_folder_id')

def vprint(*args, **kwargs):
    if os.getenv('VERBOSE', '1') == '1':
        print(*args, **kwargs)

def add_to_path(dir_path: str):
    dir_path_abs = os.path.abspath(dir_path)
    if dir_path_abs not in sys.path:
        sys.path.insert(0, dir_path_abs)
        vprint(f"Added to sys: {dir_path_abs}")
    else:
        vprint(f"Already in sys: {dir_path_abs}")

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_dir = os.path.dirname(root_dir)

download_dir = os.path.join(project_dir, "data_download")
extracted_dir = os.path.join(project_dir, "data_extracted")