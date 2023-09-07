
import matplotlib.pyplot as plt

# Though the following import is not directly being used, it is required
# for 3D projection to work with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

np.random.seed(5)

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = [
    # ("k_means_iris_8", KMeans(n_clusters=8, n_init="auto")),
    ("k_means_iris_3", KMeans(n_clusters=3, n_init="auto")),
    # ("k_means_iris_bad_init", KMeans(n_clusters=3, n_init=1, init="random")),
]

fig = plt.figure(figsize=(10, 8))

titles = [ "8 clusters", "3 clusters", "3 clusters bad init"]
titles = [ "3 clusters"]
colors = ["#FF0000", "#0000FF", "#00FF00","#111111"]
for idx, ((name, model), title) in enumerate(zip(estimators, titles)):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d", elev=48, azim=134)

    model.fit(X)
    labels = model.labels_.copy()
    labels[labels==1] = -1
    labels[labels==0] = 1
    labels[labels==-1] = 0

    print("score", model.score(X))
    # k_means_labels = model.labels_
    print("silhouette_score: ", silhouette_score(X,model.labels_ ))

    # labels = model.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor="k")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlabel("Largeur des pétales")
    ax.set_ylabel("Longueur des sépales")
    ax.set_zlabel("Longueur des pétales")
    ax.set_title(title)

# Plot the ground truth
ax = fig.add_subplot(1, 2, 2, projection="3d", elev=48, azim=134)

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X[y == label, 3].mean(),
        X[y == label, 0].mean(),
        X[y == label, 2].mean() + 2,
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor="k")

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.set_xlabel("Largeur des pétales")
ax.set_ylabel("Longueur des sépales")
ax.set_zlabel("Longueur des pétales")
ax.set_title("Données réelles")

plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.tight_layout()
plt.show()

plt.savefig('./figs/p2c4_04_iris.png')
