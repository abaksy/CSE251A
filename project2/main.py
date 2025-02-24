from dataloader import WineDataLoader
from constants import *
import logging
import sys
from baseline import BaselineModel
from model import RandomFeatureModel, CustomModel
import matplotlib.pyplot as plt
import numpy as np

MAX_ITERS = 70

plt.set_loglevel("warning")

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

test_cases = [
    ('gd', 0.5),
    ('gd', 0.2),
    ('gd', 0.1),
    ('newton', 0.0) # alpha doesnt matter for newton method
]

loader = WineDataLoader(DATASET_URL, DATASET_DIR, scale=True)

X_train, y_train = loader.load_data()

root.info("Training standard LogisticRegression Model")

baseline = BaselineModel()

L_star = baseline.fit(X_train, y_train)

root.info(f"Baseline Model - Train Loss: {L_star}")

seeds = np.random.randint(0, 100, 30)

for i, tc in enumerate(test_cases):
    plt.figure()
    wt_update, alpha = tc
    model = CustomModel(X_train, y_train, root, n_iter=MAX_ITERS, wt_update=wt_update, alpha=alpha)
    l1 = model.learn()
    
    loss_data = list()
    for j in range(30):
        model = RandomFeatureModel(X_train, y_train, root, n_iter=MAX_ITERS, wt_update=wt_update, alpha=alpha, seed=seeds[j])
        l2 = model.learn()
        if len(l2) < MAX_ITERS + 1:
            l2 += [l2[-1]] * (MAX_ITERS + 1 - len(l2))
        loss_data.append(l2)
    data = np.array(loss_data)
    
    # Calculate means for each position
    confidence = 0.95
    
    # Calculate confidence intervals
    mean_loss_random = np.mean(loss_data, axis=0)
    std_loss_random = np.std(loss_data, axis=0)
    stderr_loss_random = std_loss_random / np.sqrt(30)
    confidence_interval = 1.96 * stderr_loss_random  # 95% confidence interval

    ci_array = np.array(confidence_interval)
    # lower_bound = ci_array[:, 0]
    # upper_bound = ci_array[:, 1]

    x = range(len(mean_loss_random))
    
    # Plot mean line
    plt.plot(x, mean_loss_random, 'b-', label='Random Feature Co-ordinate Descent (Mean)', linewidth=2)
    
    # Plot confidence interval
    plt.fill_between(range(MAX_ITERS + 1), mean_loss_random - confidence_interval, 
                 mean_loss_random + confidence_interval, color='gray', alpha=0.3, label='95% CI')

    if wt_update == 'newton':
        title = "Loss Curves for Newton Update"
    else:
        title = rf"Loss Curves for First-Order Update ($\alpha = {alpha}$)"

    plt.plot(l1, label="Gradient-Based Co-Ord Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    # Add a dotted line at the value of L_star
    plt.axhline(y=L_star, color="r", linestyle="--", label="L-BFGS Optimizer")
    plt.legend()
    plt.title(title)
    plt.savefig(f"loss_{i}.png", dpi=300)
