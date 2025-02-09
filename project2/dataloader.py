import requests
import zipfile
import os
import pandas as pd
from constants import COL_NAMES


class WineDataLoader:
    def __init__(self, dataset_url, dataset_dir="dataset"):
        self.dataset_url = dataset_url
        self.dataset_dir = dataset_dir
        self.zip_path = f"{self.dataset_dir}/wine.zip"
        self.datafile_path = f"{self.dataset_dir}/wine.data"

    def download_file(self):
        # NOTE the stream=True parameter
        with requests.get(self.dataset_dir, stream=True) as r:
            r.raise_for_status()
            with open(self.zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return self.zip_path

    def extract_file(self):
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(f"{self.dataset_dir}/")

    def load_data(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        # download the file

        if not os.path.exists(self.zip_path):
            self.download_file(self.dataset_url, self.zip_path)
        # unzip the file
        if not os.path.exists(self.datafile_path):
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(f"{self.dataset_dir}/")

        df = pd.read_csv(self.datafile_path, header=None, names=COL_NAMES)
        return df
