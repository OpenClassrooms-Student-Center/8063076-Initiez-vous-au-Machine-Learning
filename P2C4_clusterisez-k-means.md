# P2C4 : Clusterisez avec k-means ()
## Découvrez le principe du clustering
  Questions abordées par le clustering: segmentation
  Le non supervisé: quand il n’y a pas de variable cible.

Jusqu'à maintenant nous avons travaillé sur la regression et la classification qui sont des techniques  d'apprentissage supervisées. Pour rappel cela signifie que le dataset d'entrainement du modele contient la valeur de la variable cible.

Dans ce chapitre, nous allons travailler sur une technique de prediction non supervisée, le regroupement ou clustering en anglais.

Le principe du clustering est de regrouper, de façon automatique, les échantillons du dataset qui se ressemblent.
On parititionne les données en un nombre fini de sous-ensemble similaires, les classes ou categories.

Pour ce qui est de la prédiction, le modèle entraîné associe à un nouvel échantillon, le sous ensemble le plus ressemblant.

L'algorithme de clustering a 2 objectifs
- minimiser  la differences entre les echantillons d'un meme  groupe. On veut des groupes denses et homogènes.
- maximiser la difference entre les differents groupes. on veut des groupes bien séparés les uns des autres.


illustrations: 3 plots:  groupes coherents collés, groupes separés mais tres dispersés; groupes coherents et separés
/figs/p2c4_01.png


Poser le probleme comme cela amene immediatement 3 questions (si si)

1. comment reconnaitre la similarité des echantillons?

Il nous faut definir une mesure de similarité pour quantifier à quel point deux échantillons sont proches ou differents.

Il existe plusieurs type de mesure de similarité, (vous vous en doutiez non ?).

Les distances couramment utilisées sont la distance euclidienne ou la distance de Manhattan. Nous verrons les 2.

2. comment trouver le nombre de clusters optimaux ?

Ce nombre est un des parametres d'entree du modele. Vous pouvez decider de partitionner les échantillons en 2, 3 ou plus de groupes. Cette decision vous appartient et depend du contexte dans lequel vous travaillez, de votre connaissance des données et de vos objectifs. Cependant, il n'est pas toujours possible de connaître à l'avance le nombre de groupes qui sera optimum.

Il nous faut donc estimer ce parametre avant d'entrainer le modele.

Pour cela, il existe plusieurs methodes aux noms poetiques de Méthode du coude (Elbow Method) ou Méthode de la silhouette (Silhouette Method). Nous comparerons les 2.

3. troisieme question, quelles types de modele pour le clustering ?
Scikit-learn en offre pas moins de 14! https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster . Nous nous contenterons d'etudier la plus simple appelé k-means()


## Quelle applications pour le clustering?

Voici quelques exemples d'applications du clustering :

- L'analyse des données géospatiales pour identifier des zones géographiques similaires en regroupant des points de données géographiques similaires. par exemple pour determoner les caracteristiqeus specifiques de certaines forets dans le contexte de la biodiversité ou de la prévention des feux.

 par exemple, dans une image aérienne ou satellitaire un SIG peut traiter différemment les forêts, champs, prairies, routes, zones humides, etc. ici considérés comme des sous-espaces homogènes.

- L'analyse de texte pour regrouper des documents similaires facilitant la compréhension et l'organisation de grands ensembles de données textuelles. Par exemple pour analyser un corpus constitué par tous les articles d'une publication pour comprendre les thèmes principaux ou recurrents

- La détection d'anomalies en regroupant les données normales dans un cluster et en considérant toute donnée non conforme comme une anomalie. Par exemple, ...

- le marketing en regroupant les clients en fonction de leurs comportements d'achat, préférences ou caractéristiques démographiques de façon à mieux les cibler.

Note: Il existe d'autres techniques non-supervisées comme la reduction de dimension mais que nous n'aborderons pas ici.


# comprendre le k-means,

L'algorithme du k-means est simple et efficace. En voici les étapes


scikit
https://scikit-learn.org/stable/modules/clustering.html#k-means


- phase d'initialisation
Tout d'abord, on choisi arbitrairememt un nombre de clusters K.

- le modèle commence par attribuer aléatoirement des points de données à ces clusters.

Initialisation : Le processus commence par sélectionner aléatoirement K points comme centres de clusters initiaux, appelés "centroïdes".

A chaque iteration
Attribution des données : Chaque point de données est attribué au cluster dont le centroïde est le plus proche en termes de distance.

Mise à jour des centroïdes : Les coordonnées des centroïdes des clusters sont recalculés en prenant la moyenne de tous les points de données appartenant à chaque cluster.

Ce processus se répète jusqu'à ce que les centres de gravité ne changent plus ou jusqu'à ce qu'un certain critère d'arrêt soit atteint. Le résultat final est une répartition des données en K clusters, chaque cluster regroupant des données similaires.


Répétition : Les étapes 3 et 4 sont répétées jusqu'à ce que les centroïdes ne changent plus de manière significative ou jusqu'à ce qu'un critère d'arrêt prédéfini soit atteint, comme un nombre maximum d'itérations.

Résultat : Une fois que le processus est terminé, vous obtenez K clusters avec des points de données similaires regroupés ensemble.


K-means est un algorithme simple pour regrouper des données, mais il nécessite de choisir judicieusement le nombre de clusters au départ et peut être sensible à l'initialisation des centroïdes.

Le resultat final va dependre de plusieurs choses
- la mesure de similarité
- le nombre de clusters choisi initialement.
- l'initialisation:

Dans la plupart des implementations du k-means, l'initialisation des premiers centroïdes se fait de façon automatisée et aléatoire. Ce caractère aléatoire implique que l'on peut obtenir des résultats sensiblement différent a chaque entrainement du modele.



## Application

Nous allons tout d'abord travailler sur un dataset artificiel que nous allons créer. Cela afin de bien comprendre la dynamique de l'agorithme et de mettre en oeuvre les techniques de calcul du nombre optimal de cluster.

Puis nous appliquerons le k-means sur des données réelles.

https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#sphx-glr-auto-examples-cluster-plot-mini-batch-kmeans-py

Pour creer un dataset artificiel en 2 dimensions constitué de K clusters de N echantillons chacun, scikit-learn offre la fonction make_blob qui est bien nommée, elle fait des blobs!

Par exemple voici 3 "blobs" bien séparés constitués de 3000 échantillons ```n_samples=3000``` chacun, centrés respectivement aux coordonnées ```centers = [[2, 2], [-2, -2], [2, -2]] ```et d'ecart type ```cluster_std=0.7``` (l'écart type dicte la dispersion des points )

from sklearn.datasets import make_blobs
centers = [[2, 2], [-2, -2], [2, -2]]
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

X est un array a 2 dimensions et labels_true un array qui contient le numéro du  cluster de chaque échantillon

Pour ne pas alourdir le texte je laisse de coté le codes de visualisation des resultats, des groupes. mais vous trouverez le code in extenso dans le notebook jupyter associé a ce chapitre.

On obtient donc 3 nuages de points bien distincts:
/figs/p2c4_01.png

Appliquons maintenant l'algorithme du k-means sur ces 3 nuages de points.

Nous utilisons le k-means de scikit-learn


from sklearn.cluster import KMeans
n_clusters = 3
k_means = KMeans(init="k-means++", n_clusters=n_clusters, random_state = 808 , n_init = 'auto'  )
k_means.fit(X)
print(k_means.cluster_centers_)

On obtient les centres suivant

[[-2.04071455 -1.99201238]
 [ 1.99552417  2.0176033 ]
 [ 1.9625611  -2.02853613]]

soit très proches des centres initiaux de nos données: [[2, 2], [-2, -2], [2, -2]]

3 facons de valuer la perfoamce

-
