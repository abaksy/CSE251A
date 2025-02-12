from dataloader import WineDataLoader
from constants import *
import logging
import sys
from baseline import BaselineModel
from model import RandomFeatureModel, CustomModel

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

loader = WineDataLoader(DATASET_URL, DATASET_DIR, scale=True)

X_train, y_train, X_test, y_test = loader.load_data()

root.info("Training standard LogisticRegression Model")

baseline = BaselineModel()

L_star = baseline.fit(X_train, y_train)

root.info(f"Baseline Model - Train Loss: {L_star}")

root.info("Method: Custom Model")

model = CustomModel(X_train, y_train, X_test, y_test, root, n_iter=50)

model.learn()

root.info("Method: Random Feature")

model = RandomFeatureModel(X_train, y_train, X_test, y_test, root, n_iter=50)

model.learn()
