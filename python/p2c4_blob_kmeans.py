# p2c4_blob_kmeans

import numpy as np
np.random.seed(808)
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt


centers = [[1, 1], [-1, -1], [1, -1]]

if False:
    colors = ["#4EACC5", "#FF9C34", "#4E9A06","#111111"]

    fig = plt.figure(figsize=(18, 6))

    groups = {
        'coh_near' : {'centers': [[0.2, 0.2], [-0.2, -0.2], [0.2, -0.2]], 'std_': 0.2, 'title': "denses et collés", 'n_samples': 3000},
        'disp_far' : {'centers': [[4, 4], [-4, -4], [4, -4]], 'std_': 1.9, 'title': "dispersés et éloignés", 'n_samples': 1500},
        'coh_far' : {'centers': [[2, 2], [-2, -2], [2, -2]], 'std_': 0.5, 'title': "denses et éloignés", 'n_samples': 3000},
    }
    n = 1
    for title, item in groups.items():
        print(title, item['std_'])

        X, labels_true = make_blobs(n_samples=item['n_samples'], centers=item['centers'], cluster_std=item['std_'])

        # coherents mais collés
        ax = fig.add_subplot(1, 3, n)
        for k, col in zip(range(len(centers)), colors):
            ax.plot(X[labels_true == k, 0], X[labels_true == k, 1], "w", markerfacecolor=col, marker="o", markersize=3, alpha = 1)
        for k, col in zip(range(len(centers)), colors):
            ax.plot(
                item['centers'][k][0],
                item['centers'][k][1],
                "o",
                markerfacecolor='#CCC',
                markeredgecolor=col,
                markersize=9,
            )
        ax.set_title(item['title'])
        n +=1
        ax.axis('off')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)


    plt.tight_layout()
    plt.show()
    plt.savefig('./figs/p2c4_01.png')

if False:

    centers = [[2, 2], [-2, -2], [2, -2]]
    X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
    n_clusters = len(centers)
    fig = plt.figure(figsize=(6, 6))
    colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

    # KMeans
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title("3 clusters")

    for k, col in zip(range(len(centers)), colors):
        my_members = labels_true == k
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker="o", markersize=4, alpha = 1)
    for k, col in zip(range(len(centers)), colors):
        ax.plot(
            centers[k][0],
            centers[k][1],
            "o",
            markerfacecolor='#CCC',
            markeredgecolor=col,
            markersize=9,
        )
    # ax.set_title("Original")
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_linestyle('dotted')
    ax.spines['left'].set_linestyle('dotted')

    # Customize the appearance of grid lines (dotted and alpha=0.5)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    plt.savefig('./figs/p2c4_02.png')

# ---

if True:

    centers = [[2, 2], [-2, -2], [2, -2]]
    X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.9)

    from sklearn.cluster import KMeans
    n_clusters=3
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, random_state = 808 , n_init = 'auto'  )
    k_means.fit(X)
    print(k_means.cluster_centers_)


    # trouver le cluster de chqaue echantillon
    k_means_labels = k_means.predict(X)
    # k_means_labels = pairwise_distances_argmin(X, k_means.cluster_centers_)


    fig = plt.figure(figsize=(12, 6))
    colors = ["#4EACC5", "#FF9C34", "#4E9A06","#111111"]
    colors = ["#FF0000", "#0000FF", "#00FF00","#111111"]
    # colors = [
    #     "#FF0000",  # Red
    #     "#FFA500",  # Orange
    #     "#FFFF00",  # Yellow
    #     "#008000",  # Green
    #     "#00FF00",  # Lime
    #     "#00FFFF",  # Cyan
    #     "#0000FF",  # Blue
    #     "#800080",  # Purple
    #     "#FF00FF",  # Magenta
    #     "#FF1493",  # Deep Pink
    #     "#8B4513",  # Saddle Brown
    #     "#696969"   # Dim Gray
    # ]

    # KMeans
    ax = fig.add_subplot(1, 2, 1)

    # for k, col in zip(range(n_clusters), colors):
    for k in range(n_clusters):
        ax.plot(X[labels_true == k, 0], X[labels_true == k, 1], "w", markerfacecolor=colors[k], marker="o", markersize=6, alpha = 1)

    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_linestyle('dotted')
    ax.spines['left'].set_linestyle('dotted')
    # ax.spines['bottom'].set_alpha(0.5)
    # ax.spines['left'].set_alpha(0.5)

    # Customize the appearance of grid lines (dotted and alpha=0.5)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Original")

    ax = fig.add_subplot(1, 2, 2)
    for k in range(n_clusters):
        cluster_center = k_means.cluster_centers_[k]
        ax.plot(X[k_means_labels == k, 0], X[k_means_labels == k, 1], "w", markerfacecolor=colors[k], marker="o", markersize=6, alpha = 1)
    for k in range(n_clusters):
        cluster_center = k_means.cluster_centers_[k]
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor='#000',
            markeredgecolor=colors[k],
            markersize=9,
        )
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_linestyle('dotted')
    ax.spines['left'].set_linestyle('dotted')
    # ax.spines['bottom'].set_alpha(0.5)
    # ax.spines['left'].set_alpha(0.5)

    # Customize the appearance of grid lines (dotted and alpha=0.5)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("KMeans")
    plt.tight_layout()
    plt.show()
