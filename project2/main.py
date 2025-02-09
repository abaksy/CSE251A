from dataloader import WineDataLoader
from constants import *

loader = WineDataLoader(DATASET_URL, DATASET_DIR, scale=False)

df = loader.load_data()

print(df.head(10), '\n', df.shape)

print(df.value_counts('label'))