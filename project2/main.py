from dataloader import WineDataLoader
from constants import *
import logging
import sys
from baseline import BaselineModel
from model import RandomFeatureModel, CustomModel
import matplotlib.pyplot as plt

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

test_cases = [
    ('gd', 0.75),
    ('gd', 0.5),
    ('gd', 0.2),
    ('gd', 0.1),
    ('newton', 0.0) # alpha doesnt matter for newton method
]

loader = WineDataLoader(DATASET_URL, DATASET_DIR, scale=True)

X_train, y_train, X_test, y_test = loader.load_data()

root.info("Training standard LogisticRegression Model")

baseline = BaselineModel()

L_star = baseline.fit(X_train, y_train)

root.info(f"Baseline Model - Train Loss: {L_star}")


for i, tc in enumerate(test_cases):
    plt.figure()
    wt_update, alpha = tc
    model = CustomModel(X_train, y_train, X_test, y_test, root, n_iter=50, wt_update=wt_update, alpha=alpha)
    l1 = model.learn()
    model = RandomFeatureModel(X_train, y_train, X_test, y_test, root, n_iter=50, wt_update=wt_update, alpha=alpha)
    l2 = model.learn()
    if wt_update == 'newton':
        title = "Loss Curves for Newton Update"
    else:
        title = rf"Loss Curves for First-Order Update ($\alpha = {alpha}$)"

    plt.plot(l1, label="Gradient-Based Co-Ord Descent")
    plt.plot(l2, label="Random Feature Co-Ord Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    # Add a dotted line at the value of L_star
    plt.axhline(y=L_star, color="r", linestyle="--", label="L-BFGS Optimizer")
    plt.legend()
    plt.title(title)
    plt.savefig(f"loss_{i}.png", dpi=300)
