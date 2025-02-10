import requests
import zipfile
import os
import pandas as pd
import numpy as np
from constants import COL_NAMES
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class WineDataLoader:
    def __init__(self, dataset_url, dataset_dir="dataset", train=0.75, scale=True):
        assert train < 1

        self.dataset_url = dataset_url
        self.dataset_dir = dataset_dir
        self.zip_path = f"{self.dataset_dir}/wine.zip"
        self.datafile_path = f"{self.dataset_dir}/wine.data"
        self.train_ratio = train
        self.scale = scale

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

    def scale_data(self, df):
        """
        Scale columns of df and return it
        """
        scaler = StandardScaler()
        df[COL_NAMES[1:]] = scaler.fit_transform(df[COL_NAMES[1:]])
        return df

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

        # Filter first two classes from dataset
        df = df[df["label"].isin([1, 2])]

        # replace 2 with 0 in label column
        df["label"] = df["label"].replace(2, 0)

        # Normalize features of the dataset
        if self.scale:
            df = self.scale_data(df)

        N = df.shape[0]
        print(df.iloc[:, 1:].to_numpy().shape)
        X = np.hstack((np.ones(N).reshape(-1, 1), df.iloc[:, 1:].to_numpy()))
        y = df["label"]

        return X, y
