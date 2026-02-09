# File to get datasets from Google Drive

from dotenv import load_dotenv
import os

load_dotenv()
my_google_key = os.environ['google_key']
data_folder = os.environ['default_data_folder']


def load_data(prefix=None):
    """
    Function to load data samples from Google Drive
    
    :param prefix: Description
    """
    # conda install pytorch::pytorch
    pass