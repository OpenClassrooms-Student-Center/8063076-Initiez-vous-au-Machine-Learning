# !pip install palmerpenguins
# %pip install palmerpenguins

from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
# sns.set_style('whitegrid')
penguins = load_penguins().sample(frac = 1, random_state = 808)


colors = ["#FF0000", "#0000FF", "#00FF00","#111111"]

for col in ['species', 'island','sex']:

    k = 0
    for value in penguins[col].unique():
        penguins.loc[penguins[col] == value, col] = k
        k +=1

penguins.dropna(inplace = True)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# print("-- PCA")
# X = pca.fit_transform(penguins[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']])

print("-- raw")
X = penguins[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']].values
y = penguins.species.values

model = KMeans(n_clusters=3, init='random', max_iter=100, random_state=101, n_init = 'auto')
model.fit(X)
print("score", model.score(X))
labels = model.labels_.copy()
labels[labels==1] = -1
labels[labels==0] = 1
labels[labels==-1] = 0
print("silhouette_score: ", silhouette_score(X,k_means_labels ))


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
# print("-- PCA")
X = pca.fit_transform(penguins[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']])

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
for k, col in zip(range(3), colors):
    ax.plot(X[y == k, 0], X[y == k, 1], "w", markerfacecolor=col, marker="o", markersize=6, alpha = 1)

ax = fig.add_subplot(1, 2, 2)
for k, col in zip(range(3), colors):
    ax.plot(X[k_means_labels == k, 0], X[k_means_labels == k, 1], "w", markerfacecolor=col, marker="o", markersize=6, alpha = 1)

plt.tight_layout()
plt.show()

# regardons la matrice de confusion

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print("confusion_matrix",confusion_matrix(list(y),k_means_labels ))
print("accuracy_score",accuracy_score(list(y),k_means_labels ))
