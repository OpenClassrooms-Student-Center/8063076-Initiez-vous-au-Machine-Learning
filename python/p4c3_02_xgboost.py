import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


X, y = make_hastie_10_2(n_samples=4000, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

learning_rates = [1, 0.6,  0.3, 0.1]


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1, 1, 1)

for lr in learning_rates:

    clf = ensemble.GradientBoostingClassifier(
                    n_estimators= 500,
                    max_depth= 2,
                    random_state= 8,
                    learning_rate= lr
    )
    clf.fit(X_train, y_train)

    scores = np.zeros((clf.n_estimators,), dtype=np.float64)
    for i, y_proba in enumerate(clf.staged_predict_proba(X_test)):
        scores[i] =  log_loss(y_test, y_proba[:, 1])

    ax.plot(
        (np.arange(scores.shape[0]) + 1),
        scores,
        "-",
        label=f"alpha: {lr}",
    )

ax.grid(True, which = 'both')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel("Itérations")
ax.set_ylabel("Log Loss  (test)")
ax.set_title("Influence du learning rate sur la performance du GradientBoosting")
ax.legend()
plt.tight_layout()
plt.show()
plt.savefig("./figs/p4c3_04_learning_rate.png")
