from dataloader import WineDataLoader
from constants import *
import logging
import sys
from baseline import BaselineModel
from model import RFLogisticRegression

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

loader = WineDataLoader(DATASET_URL, DATASET_DIR, scale=True)

X_train, y_train = loader.load_data()

root.info("Training standard LogisticRegression Model")

baseline = BaselineModel()

L_star = baseline.fit(X_train, y_train)

root.info(f"Baseline Model - Loss: {L_star}")

model = RFLogisticRegression(X_train, y_train, root)

model.learn()