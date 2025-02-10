from dataloader import WineDataLoader
from constants import *

loader = WineDataLoader(DATASET_URL, DATASET_DIR, train=0.8, scale=False)

df_train, df_test = loader.load_data()

print(df_train.shape, df_test.shape)