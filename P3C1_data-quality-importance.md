# Privilégiez une approche data-centric

    importance de la data; data centric AI vs model centric: les gains de performance sont principalement obtenus en travaillant sur la data et moins sur la paramétrisation du modèle.
    revenir sur l'importance de la robustesse du modèle: le modèle doit performer sur des données absentes du jeu d'entraînement (mais similaires). C'est le but principal d'un projet ML.

# Trouvez des jeux de données pour le machine learning
    Sources de data:
    Les références: site UCI, kaggle, google data search, portails data opensources des villes, pays + EU.
    Créer son dataset avec sklearn permet de tester la pertinence de son approche. exemples illustrés pour de la classification et du clustering sur des dataset non facilement séparables.

# À vous de jouer !
    Créez un jeu de données pour un problème donné en utilisant sklearn

# Comprenez la différence entre corrélation et causalité
    corrélation ne veut pas dire causalité: pas parce qu’on a de la data qu’on peut avoir un modèle. autrement dit: y a-t-il du signal dans les données?

# Anticipez les impacts du biais dans les données d'entraînement
    Attention au biais dans la data: bias in, bias out. Exemples de données d'entraînement biaisées



On peut approcher la construction de modèles performants de 2 façons suivant selon que l'on se concentre sur les modèles ou les données.

L'approche model-centric est la plus courante et la plus enseignée. On vous fournit un dataset explicite et propre, à vous de créer un modèle performant.

Dans le monde réel, choisir un modèle et l'optimiser par rapport à un dataset donné est un problème résolu. Les classes de modèles les plus efficaces sont connus et disponibles. Soit sous forme open source (via des librairies de type scikit-learn, tensorflow, etc ...) soit via des services clouds d'auto ML: Vertex AI, Sagemaker, etc ...

Les gains de performances ne vont donc pas se faire sur votre aptitude à optimiser le modèle que vous aurez choisi mais sur la façon dont vous allez élaborer le jeu de données d'entraînement. C'est l'approche data-centric, introduite par Andrew NG en 2001.
https://www.youtube.com/watch?v=TU6u_T-s68Y


Cette approche centrée qui se concentre plus sur les données que sur le modèle recouvre une série de pratiques pour booster les performances du modèle en manipulant les données.

L'idée principale sera d'être attentif
- à la qualité du jeux de données: valeurs manquantes, outliers, mauvais étiquetage, biais de représentation, ...
- aux erreurs du modèle. Comprendre pourquoi certains échantillons posent problème et transformer ces caractèristiques en variables que le modèle puisse apprendre.


Dans cette troisième partie du cours, nous allons donc nous concentrer sur la partie données du couple données - modèle.

## Trouver des sources de données.

Avant de nous attacher à améliorer la qualité ou comment transformer un jeux de données, regardons où l'on peut trouver des jeux de données.

- en 2018, Google lanca son moteur de recherche dédié aux jeux de données: https://datasetsearch.research.google.com/
- kaggle la plateforme de competition de machine learning, offre aussi de nombreux datasets sur https://www.kaggle.com/datasets
- Nous avons deja vu les sites dédiés au ML comme UCI: https://archive.ics.uci.edu/
- les sites des institutions ont souvent une politique d'open source. on citera la ville de Paris https://opendata.paris.fr/pages/home/, de Londres https://data.london.gov.uk/dataset, de Rome https://dati.comune.roma.it/, les institutions européennes https://data.europa.eu/en,
- les agences scientifiques : ademe https://data.ademe.fr/, EdF https://opendata.edf.fr/pages/welcome/, biodiversité https://globil-panda.opendata.arcgis.com/ et https://www.gbif.org/
- BigQuery, un service de big data de Google Cloud met à disposition gratuitement des datasets extremement interessants. https://cloud.google.com/bigquery/public-data?hl=fr

- et enfin les librairies elles mêmes
	- scikit learn met a disposition des datasets simples (toy dataset) ou plus complexes (real world)
    	https://scikit-learn.org/stable/datasets/toy_dataset.html
    	https://scikit-learn.org/stable/datasets/real_world.html
	- mais statsmodel https://www.statsmodels.org/devel/datasets/index.html#available-datasets
	les librairies R https://vincentarelbundock.github.io/Rdatasets/articles/data.html etc ...

Vous avez donc le choix pour vous familiariser avec des types de données et des taches machine learning variées.

## Creer son propre dataset

Dans le chapitre sur le clustering, nous avons construit nous même un dataset de travail propre au problème de partitionnement automatique.

Scikit-learn offre d'autres méthodes de constructions de dataset qui sont utiles pour des tâches de classification, régression ou de clustering. Nous avons vu make_blobs pour le clustering mais il y a aussi make_régression et make_classification pour creer des datasets adaptés à la regression et à la classification.

D'autres méthodes https://scikit-learn.org/stable/modules/classes.html#samples-generator permettent de générer des données plus complexes, notamment qui ne soient pas linéairement séparables. On ne peut pas tracer une droite séparant les différentes catégories de nuages de points).

img/p3c1-make-manifold-01.png
img/p3c1-make-manifold-02.png

# A vous de jouer

Regardez la methode make_classification https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification et générez un dataset avec les caractéristiques suivantes:

- classification binaire (n_classes)
- 1000 échantillons (n_samples)
- 3 variables (n_features) toutes 2 utiles (n_informative, n_redundant)


N'oubliez pas de donner une valeur à random_state pour pouvoir reproduire vos expériences.

Ensuite, entraînez une simple régression logistique sur tout le dataset et observer les performances du modèle.

Plus challenging pour la régression logistique, utilisez make_circles https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html et make_moons https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons pour générer des datasets de la forme

img/p3c1-make-circle.png
img/p3c1-make-moons.png

puis entraînez une régression logistique et observez la performance du modèle. Elle devrait chuter car les nuages des échantillons des différentes catégories ne sont pas linéairement séparables.

Vous pouvez générer un dataset de type moon avec par exemple :

data = datasets.make_moons(n_samples=n_samples, noise=0.05)
et de type circle (cercle) avec
data = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

en retour vous avez
X = data[0] l'array numpy de dimensions 2 des predicteurs
et y = data[1] l'array numpy des catégories de chaque échantillon.

# le biais dans les données

L'IA est maintenant présente dans tous les secteurs.
Dans ce contexte, il nous faut nous assurer que les modèles que nous construisons ne sont pas biaisés.

Un modèle biaisé est un modèle dont les prédictions sont systématiquement distordues. Avec pour conséquence un risque de décisions systématiquement inéquitables ou inexactes.

La première cause de biais sont le fait d'un mauvais échantillonnage des données d'entraînement. Par exemple, si un sondage d'opinion n'interroge que les utilisateurs d'iphone les résultats ne reflèteront surement pas l'opinion de toute la population.

Plus concretement.

- En ressource humaine, quand les données historiques d'entraînement contiennent plus d'hommes que de femmes pour un poste donnée (ou vice versa), un modèle de présélection de candidats aura tendance à favoriser les profils d'hommes pour ce poste. D'ou une discrimination avérée bien qu'involontaire.
- Dans le cadre bancaire, un modèle entraîné principalement sur des fraudes en ligne, sera incapable de detecter d'autre type de fraudes (en personne ou internes).

Néanmoins le biais n'est pas toujours dû à une sous représentation d'une catégorie d'évenement.

Dans le cas des LLMs (Large Language Models) de type chatGPT ou Bard, le corpus d'entraînement comporte des biais de représentations qui peuvent être historiques, culturels, géographiques ou dus à des stéréotypes. Le modèle aura alors tendance à reproduire les représentations rencontrées pendant son apprentissage.

Pour atténuer le biais, il faut donc s'assurer de l'exhaustivité des valeurs potentielles des variables prédictrices ou cibles et mettre en place des stratégies de remédiations. En ce qui concerne les LLMS, vos prompts sont automatiquement enrichis d'instructions permettant de limiter le biais intrinsèque des modèles.
