from dataloader import WineDataLoader
from constants import *

loader = WineDataLoader(DATASET_URL, DATASET_DIR)

df = loader.load_data()

print(df.head(10), '\n', df.shape)