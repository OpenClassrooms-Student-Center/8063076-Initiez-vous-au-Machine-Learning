# P2C4 : Clusterisez avec k-means ()
## Découvrez le principe du clustering

Jusqu'à maintenant, nous avons travaillé sur la des techniques d'apprentissage supervisées comme la régression ou la classification. Le dataset d'entraînement du modèle contient la valeur de la variable cible.

Dans ce chapitre, nous allons travailler sur une technique de prédiction non supervisée, le partitionnement ou clustering en anglais.

Le principe du clustering est de regrouper, de façon automatique, les échantillons du dataset qui se ressemblent.
On partitionne les données en un nombre fini de sous-ensemble similaires que l'on appelle groupes, classes ou catégories.

L'algorithme de clustering a 2 objectifs:

- minimiser la différence entre les échantillons d'un même groupe. On veut des groupes denses et homogènes.
- maximiser la différence entre les groupes. On veut des groupes bien séparés les uns des autres.


illustrations: 3 plots: groupes cohérents collés, groupes séparés mais très dispersés; groupes cohérents et séparés
/figs/p2c4_01.png


Poser le problème comme cela amène immédiatement 3 questions (si si 3)

1. Comment reconnaître la similarité des échantillons?


La mesure utilisée par défaut et donc la plus usitée est la distance euclidienne. La différence, au carré, entre 2 points.

2. comment trouver le nombre de clusters optimaux ?

C'est un des paramètres d'entrée du modèle. Vous pouvez décider de partitionner les échantillons en 2, 3 ou plus de groupes. Cette décision vous appartient et dépend du contexte dans lequel vous travaillez, de votre connaissance des données et de vos objectifs. Cependant, il n'est pas toujours possible de connaître à l'avance le nombre de groupes qui sera optimum.

Il nous faut donc estimer ce paramètre avant d'entraîner le modèle.

Pour cela, il existe plusieurs méthodes aux noms poétiques de Méthode du coude (Elbow Method) ou Méthode de la silhouette (Silhouette Method).

3. troisième question, quels types d'algorithme pour le clustering ?

Scikit-learn en offre pas moins de 14! https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster . Nous nous contenterons d'étudier la plus simple appelé k-means.


## Quelles applications pour le clustering?

Voici quelques exemples d'applications du clustering :

- L'analyse des données géospatiales pour identifier des zones géographiques similaires en regroupant des points de données géographiques similaires. par exemple pour déterminer les caractéristiques spécifiques de certaines forêts dans le contexte de la biodiversité ou de la prévention des feux.

 Par exemple, dans une image aérienne ou satellitaire un SIG peut traiter différemment les forêts, champs, prairies, routes, zones humides, etc. considérés comme des sous-espaces homogènes.

- L'analyse de texte pour regrouper des documents similaires facilitant la compréhension et l'organisation de grands ensembles de données textuelles. Par exemple pour analyser un corpus constitué par tous les articles d'une publication pour comprendre les thèmes principaux ou récurrents

- La détection d'anomalies en regroupant les données normales dans un cluster et en considérant toute donnée non conforme comme une anomalie. Par exemple, ...

- le marketing en regroupant les clients en fonction de leurs comportements d'achat, préférences ou caractéristiques démographiques de façon à mieux les cibler.

Note: Il existe d'autres techniques non supervisées comme la réduction de dimension mais que nous n'aborderons pas ici.

Le clustering est utilisé lorsque l'on veut comprendre le dataset sans a priori préalable.

# comprendre le k-means,

L'algorithme du k-means est simple et efficace. En voici les étapes

scikit
https://scikit-learn.org/stable/modules/clustering.html#k-means


- phase d'initialisation
Tout d'abord, on choisit arbitrairement un nombre de clusters K.

- le modèle commence par attribuer aléatoirement des points de données à ces clusters.

Initialisation : Le processus commence par sélectionner aléatoirement K points comme centres de clusters initiaux, appelés "centroïdes".

A chaque iteration
Attribution des données : Chaque point de données est attribué au cluster dont le centroïde est le plus proche en termes de distance.

Mise à jour des centroïdes : Les coordonnées des centroïdes des clusters sont recalculées en prenant la moyenne de tous les points de données appartenant à chaque cluster.

Ce processus se répète jusqu'à ce que les centres de gravité ne soient plus ou jusqu'à ce qu'un certain critère d'arrêt soit atteint. Le résultat final est une répartition des données en K clusters, chaque cluster regroupant des données similaires.


Répétition : Les étapes 3 et 4 sont répétées jusqu'à ce que les centroïdes ne changent plus de manière significative ou jusqu'à ce qu'un critère d'arrêt prédéfini soit atteint, comme un nombre maximum d'itérations.

Résultat : Une fois que le processus est terminé, vous obtenez K clusters avec des points de données similaires regroupés ensemble.


K-means est un algorithme simple pour regrouper des données, mais il nécessite de choisir judicieusement le nombre de clusters au départ et peut être sensible à l'initialisation des centroïdes.

Le résultat final va dépendre de plusieurs choses
- la mesure de similarité
- le nombre de clusters choisi initialement.
- l'initialisation:

Dans la plupart des implémentations du k-means, l'initialisation des premiers centroïdes se fait de façon automatisée et aléatoire. Ce caractère aléatoire implique que l'on peut obtenir des résultats sensiblement différents à chaque entraînement du modèle.



## Application

Nous allons tout d'abord travailler sur un dataset artificiel que nous allons créer. Cela afin de bien comprendre la dynamique de l'algorithme et de mettre en œuvre les techniques de calcul du nombre optimal de cluster.

Puis nous appliquerons le k-means sur des données réelles.

https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#sphx-glr-auto-examples-cluster-plot-mini-batch-k means-py

Pour créer un dataset artificiel en 2 dimensions constitué de K clusters de N échantillons chacun, scikit-learn offre la fonction make blob qui est bien nommée, elle fait des blobs!

Par exemple voici 3 "blobs" bien séparés constitués de 3000 échantillons ```n_samples=3000``` chacun, centrés respectivement aux coordonnées ```centers = [[2, 2], [-2, -2], [2, -2]] ```et d'écart type ```cluster_std=0.7``` (l'écart type dicte la dispersion des points )

from sklearn.datasets import make_blobs
centers = [[2, 2], [-2, -2], [2, -2]]
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

X est un array à 2 dimensions et labels_true un array qui contient le numéro du cluster de chaque échantillon

Pour ne pas alourdir le texte je laisse de côté le codes de visualisation des résultats, des groupes. mais vous trouverez le code in extenso dans le notebook jupyter associé à ce chapitre.

On obtient donc 3 nuages de points bien distincts:
/figs/p2c4_01.png

Appliquons maintenant l'algorithme du k-means sur ces 3 nuages de points.

Nous utilisons le k-means de scikit-learn


from sklearn.cluster import KMeans
n_clusters = 3
k_means = KMeans(init="k-means++", n_clusters=n_clusters, random_state = 808 , n_init = 'auto' )
k_means.fit(X)
print(k_means.cluster_centers_)

On obtient les centres suivant

[[-2.04071455 -1.99201238]
 [ 1.99552417 2.0176033 ]
 [ 1.9625611 -2.02853613]]

soit très proches des centres initiaux de nos données: [[2, 2], [-2, -2], [2, -2]]

Néanmoins comment évaluer la performance du modèle lorsque l'on ne connaît pas les coordonnées des centroïdes initiaux.

Il y a pour cela 3 méthodes
- calculer la distance entre les points et leur centroïdes respectifs. Un bon clustering doit regrouper au maximum les points et donc cette distance doit être minimum.

Dans scikit learn, le score du modèle est repose sur la somme des distances des points à leur centroïdes.

Ce qui donne pour l'entraînement ci dessus
k_means.score()

score: -4577.71

Comme c'est un score relatif, il sert surtout à comparer différents modèles.

- si on a un dataset pour lequel on connaît le groupe de référence, on peut utiliser les techniques d'évaluation de la performance présentées au chapitre sur la classification: matrice de confusion, précision etc ...

- graphique: en observant les nuages de points et leur répartition on peut estimer visuellement la pertinence du clustering

Pour notre modèle, on obtient
/figs/p2c4_03.png

On peut remarquer que les nuages de points sont très similaires. Seuls ont changé de clusters quelques points au-dessus de l'abscisse entre les clusters en haut et en bas droite. Le modèle est a priori plutôt bon.

- et enfin on peut utiliser le coefficient de silhouette.
La technique de la silhouette prend en compte non seulement la densité des clusters mais aussi l'écart entre les différents clusters.

Plus précisement:
Le coefficient de silhouette est calculé à partir de la distance moyenne intra-groupe (a) et de la distance moyenne entre les groupes les plus proches (b) pour chaque échantillon.
- a est donc la moyenne des distances des échantillons avec leur centroïdes respectifs dans chaque groupe
- b est la distance entre un échantillon et le groupe le plus proche dont l'échantillon ne fait pas partie.
- Le coefficient de silhouette d'un échantillon est alors défini par

silhouette = (b - a) / max(a, b).

Le coefficient de silhouette est compris entre -1 (mauvais) et 1 (excellent) .

En python, on utilise la methode silhouette_score de scikit-learn.

from sklearn.metrics import silhouette_score
k_means_labels = k_means.predict(X)
print("silhouette_score: ", silhouette_score(X,k_means_labels ))

On obtient

silhouette: 0.60

Vu la qualité de l'identification des différents clusters, observé sur le graph ci dessus, on aurait pu s'attendre à un meilleur score, plus proche de 1. Cependant le coefficient de silhouette prend aussi en compte l'aspect séparation des clusters. et il est vrai que les clusters du dataset sont assez proches les uns des autres.

Dans cet exemple, nous sommes dans un cas idéal. Nous connaissons les données, leurs groupes respectifs et le nombre de clusters.
Et en plus le dataset est linéairement séparable. On peut tracer une ligne droite de séparation entre les différents nuages de points.

En réalité, il est rare que nous connaissions à l'avance le nombre de clusters.
Dans ce cas on peut aussi utiliser le coefficient de silhouette pour déterminer le nombre optimal de clusters.

Faisons varier le nombre de clusters comme paramètres de k-means de 2 à 10 et regardons l'évolution du coefficient de silhouette.

/figs/p2c4_03.png

On observe bien un pic pour 3 clusters

# sur des données réelles

Oublions maintenant nos blobs et travaillons sur un jeu de données réelles.
S'il y a un dataset incontournable en data science, c'est Iris.
Iris est constitué de 150 échantillons de fleurs réparties en 3 familles de 50 échantillons chacune:
- Iris-Setosa
- Iris-Versicolour
- Iris-Virginica

Les variables sont:
- sepal length in cm
- sepal width in cm
- petal length in cm
- petal width in cm


Une image vaut plus qu'une explication sur la différence entre Sepal et Petal.

Le dataset iris est disponible directement dans les datasets de scikit-learn. On le charge de la façon suivante

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

ou X comprend les valeurs des prédicteurs sous forme de numpy array et y la liste des classes 0, 1 ,2.

On applique donc k-means sur ce dataset

model = KMeans(n_clusters=3, n_init="auto")
model.fit(X)

Voici les scores de classification
print("score", model.score(X))
# k_means_labels = model.labels_
print("silhouette_score: ", silhouette_score(X,model.labels_ ))

Comme on sait que les fleurs appartiennent à 3 familles, on peut aussi regarder la performance du clustering en tant qu'outil de classification. Est ce que le modèle retrouve bien les catégories initiales?
Notez bien que contrairement à une classification supervisée, ce n'est pas le but du modèle. Le modèle de clustering ne tend qu'à regrouper les échantillons qui ne ressemblent pas à détecter à quelle catégorie arbitraire ils appartiennent.

Avant cela il faut s'assurer que les catégories trouvées par le modèle de clustering correspondent bien aux catégories initiales:
En regardant les catégories du modèle, on s'aperçoit qu'il faut échanger les catégories 0 et 1.
labels = model.labels_.copy()
labels[labels==1] = -1
labels[labels==0] = 1
labels[labels==-1] = 0

Attention cet échange de catégorie dépend entièrement des paramètres du modèle et de l'ordre des échantillons dans le dataset. Il n'est pas universel.

Ceci dit, regardons le score de précision et la matrice de confusion

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("accuracy_score",accuracy_score(y,labels ))
print("confusion_matrix\n",confusion_matrix(y,labels ))

accuracy_score 0.8866666666666667
confusion_matrix
 [[50 0 0]
 [ 0 47 3]
 [ 0 14 36]]

Donc une performance assez bonne. La précision est de 88,6%
et la matrice de confusion montre que la catégorie 1 est identifié à 100% et les catégories 2 et 3 le sont en majorité

L'analyse graphique confirme cela

./figs/p2c4_04_iris.png
