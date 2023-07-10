# sections
## Comprenez l’intérêt du Machine Learning
    __Ce n’est plus une nouvelle le ML est partout. Quelques exemples frappants du ML au quotidien. ML par rapport à la Data Science et à l'IA. Motivation du cours, motivation de l'apprenant et but du projet de machine learning.__

Sans pour cela remonter au Turque méchanique du 17e siècle, on peut dire que
l'intelligence artificielle (IA) est née en 1957 avec l'invention du __Perceptron__, première machine rudimentaire de classification automatique d'images.
S'ensuivi entre 1960 et 2010 une série de printemps et d'hivers de l'IA en fonction des progrès réalisés et de l'engouement du grand public pour le domaine.
Ce n'est qu'avec l'explosion dans les dernières années de la puissance de calcul et de stockage des ordinateurs que l'IA a vraiment pris racine dans notre quotidien.

On parle aujourd'hui, d'IA citoyenne car nous avons à notre disposition des jeux de données, des librairies de machine learning puissantes et bien documentées (scikit-learn), des plateformes de travail en ligne (google colab) et multitudes de cours et tutoriaux. C'est une époque  excitante pour l'IA.

Toutefois, IA est un terme générique qui couvre aussi bien des visions de fin du monde que l'ingénierie logicielle de pointe.
Dans ce cours, nous parlerons uniquement de machine learning ou apprentissage automatique.

- Le ML est la brique de base de l'IA, la chaîne dans le vélo, le moteur dans la voiture, le rouage de l'automate
- Le ML a une tache précise à accomplir: la **prédiction**.
- Le ML utilise des **algorithmes** pour développer des modèles prédictifs à partir de **jeux de données**.

Cette fonction prédictive est polymorphe et s'appelle selon les besoins aura pour nom classification, supervision, détection, proposition, ranking, prévision ... mais à la base il y a toujours un but de prédiction.

Nous profitons de cette profusion de modèles prédictifs: proposition de contenu sur les plateformes, prévention et surveillance globale, évaluation des risques, IA générative, optimisation des chaînes de production et de ventes, détection des anomalies, prévisions temporelles etc etc

Dans ce cours nous allons travailler sur le machine learning dit classique. Le ML classique se nourrit de jeux de données tabulaires, de taille raisonnable (< 1Gb) et disponibles dans un fichier csv. On va donc laisser, les données plus lourdes, de type audio, images, vidéo, et l'IA générative au deep learning et au réseaux de neurones.

A la base du ML, se trouve le jeu de données ou dataset. Sans data, pas de ML!

## Découvrez les 2 approches en modélisation: stats vs ML;
    Il y a différentes façons d’approcher un problème concrétisé par un jeu de données.
    exemple avec dataset school (age,  poids, taille, G/F…)
    Bien comprendre la différence entre une approche de modélisation statistique et une approche Machine Learning (Pattern recognition et modèle de prédiction ou analyse de  la dynamique entre les variables et modélisation statistique)
    Conséquence directe des 2 approches : utiliser toute la donnée disponible ou en garder une partie pour l'évaluation du modèle.
    Modèle black box ou explicatif; sklearn ou statsmodel

Imaginons que  vous soyez responsable des données d'une plateforme de contenu en ligne dont le business modèle repose sur le nombre  d'abonnements souscrits. On vous pose 2 questions

- Quel est le profil des utilisateurs qui s'abonnent?
- comment prédire si un nouvel utilisateur va s'abonner ou non
Dans les 2 cas, on cherche a en savoir plus sur l'acte d'abonnement. variable cible binaire (oui / non).

Cependant, ces questions imposent des approches différentes: **modélisation statistique** contre **modélisation prédictive**.

Dans la question relative au profil, on cherche à comprendre l'acte d'abonnement en fonction des caractéristiques des utilisateurs.
On est dans une démarche d'analyse et d'interprétation de la dynamique entre les variables. Cette exploration statistique cherche à expliciter les relations entre les variables dans un but analytique. la modélisation statistique s'appuie sur des tests d'hypothèses et des modèles mathématiques pour évaluer les conclusions de l'analyse.


Dans la question relative à la prédiction, on ne cherche qu'à prédire efficacement l'acte d'abonnement. C'est l'approche machine learning.  L'important est la justesse des prédiction. Le modèle est assimilé à une boite noire. La compréhension des predictions vient dans un second temps.
On attend 2 choses du modèle predictif
1. Des prédictions de qualité, une bonne performance que l'on va evaluer par une métrique choisie au préalable
2. la capacité a extrapoler: soit à généraliser ses prédictions à partir des données d'entraînement. On parle de robustesse du modèle face a de nouveaux échantillons.

Mesurer la performance du modele implique  d'avoir defini au préalable une métrique d'évaluation. Cette metrique va dependre des donnees et de la tache a realiser.

Bien que l'on puisse utiliser les modeles statistiques pour de la prediction, la puissance des modeles ML est sans commune mesure.

tableau recap

stats
- expliquer, analyser
- tous les echantillons
- statsmodel

ML
- predire,
- etre capable de generaliser a de nouvelles données:
- performance, resilience et robustesse du modele.
- une partie des echantillon est reservé a l'evaluation des performance du  modele
- scikit-learn



## Comprenez en quoi consiste le Machine Learning
    Construire un modèle prédictif consiste à l'entraîner sur des données, puis évaluer sa performance et enfin l’optimiser.
    Attentes: le modèle prédictif doit être capable d'être performant sur des données nouvelles, c’est la capacité de généralisation, d'extrapolation  

    concepts: entraînement, évaluation, optimisation, généralisation,

Le but du ML est donc d'entraîner, un modele prédictif à partir d'un jeu de données. Mais en pratique comment cela se passe t il?

Encart: Dans la suite quand on parle de jeux de donnée, pensez a une feuille de type google spreadsheet ou tableur excel. Les variables sont les colonnes, et les échantillons sont les lignes. On distingue la variable cible, sujet de la prédiction, des autres variables potentiellement prédictrices.

On part d'un jeu de données que l'on suppose exploitable. Concept fourre tout qui suppose que les variables aient une relation entre elles. Corrélation, causalité, ... le dataset n'est pas un regroupement totalement aléatoire de données qui auraient été collectées on ne sait comment.

Prenons un exemple avec un jeux de données comprenant l'âge, la taille et le poids d'une centaine de collégiens. on suppose raisonnablement que la taille et l'âge d'un enfant sont liées à son poids.

Voici les étapes pour développer un modèle prédictif à partir de ce dataset.

1.  le travail sur la donnée:
Tout commence par un travail de transformation des données brutes pour les rendre compatibles avec le modele de ML choisi: nettoyage, normalisation, numerisation etc ... On parle de data cleansing.

Ensuite, on decoupe le dataset en 2.
Une parti des échantillons (train) sont reservés àl'entrainement du modele tandis que l'autre partie (test) est mise de coté pour évaluer sa performance sur des données qu'il n'a pas vu.   On va d'ailleurs repeter ce decoupage plusieurs fois et de façon aléatoire pour s'assurer que le modele performe dans tous les cas de répartition train / test. Cette methode s'appelle la validation croisée.

En parallele, on va chercher a optimiser la perf du modele en modifiant ses parametres et en observant son scoresur chaque version train / test du dataset.


## Distinguez l’ approche supervisée de l’approche non-supervisée
    Approches  supervisées ou non-supervisées;
    Définition d'une variable cible;
    On évalue les modèles différemment selon que l'on soit en supervisé ou non-supervisé

Un distinction importante entre 2 types de ML classique sur des données tabulaires.

soit on essaie de predire une des variables du dataset, soit on pense que les donneees appartiennnt a plusieurs groupes et on va predire a quelle groupe l'echantillon appartient.

approche supervisée: je predis le prix de location de l'appartement en fonction de sa superficie, localisation, etage etc
approche non supervisée, je repartie les appartements en N groupes distinct et le modele associe un appart a un groupe
une autre distinction dans l'approche supervisée vient de la nature de la variable cible que l'on veut predire
si la variable est un chiffre (prix, taille, emission, ...) (on dit qu'elle est continue), alors on parlera de regression
si la variable cible est une categorie, on parle de classification
on parle de classification binaire en face d'un choix binaire: 0 ou 1 , survivra ou non, achete ou non
de classification multi class qund on a plus de 2 categories: on distngue alors la natre ordonnée ou non d'un ensemble: jeune, adulte, vieux ou
peu moyen beaucoup => ordinal
d'une classification multiclass non ordonnée: chat, chien, mouton

On a parlé de variable cible. En fait ce n'est pas toujours le cas et il faut distinguer en ML l'approche supervisée de l'approche non supervisée

Imaginez que vous avez un ensemble de photos de chats et de chiens. Vous souhaitez les classer automatiquement en utilisant un modele de ML.
Dans le ML supervisé, il faut que chaque image de votre  dataset soit etiquetée en chat ou en chien. cette etiquette est la variable cible que vous allez chercher a predire. Cette etiquetage est le plus realisé par un humain et peut prendre beaucopp de temps et necessiter beaucoup de resources sur des jeux de donnees avec  beaucoup d'echantillons. On parle de processus supervisé car le modèle apprend à partir d'exemples étiquetés fournis par un humain.

En revanche, dans l'approche non supervisé, vous n'avez pas d'étiquettes pour les photos. Le modele va essayer de regrouper automatiquement les photos en fonction de leurs caractéristiques similaires. Il regroupe les en grappes / cluster sans aucune connaissance préalable sur les espèces
On parle alors de clustering.

En fin dans l'approche supervisé en fonction de la nature de la variable cible, on parlera de
- classification lorsqu'on predit des categories: binaire (oui/non), ordinal (petit, moyen, grand) ou nominales (chat - chien - brebis , bleu-rose-vert-blanc, ...)
- et de regression lorsque l'on predit une variable continue: prix, âge, salaire, volume, temperature etc

En résumé, le machine learning supervisé nécessite des données étiquetées pour entraîner le modèle à prédire des valeurs spécifiques, tandis que le machine learning non supervisé cherche à trouver des structures ou des groupements inhérents aux données sans avoir d'informations spécifiques à prédire.

on a donc 3 types de ML classique
- supervisé: regression vs classification
- non supervise: clustering

## À vous de jouer !
    Aller sur UCI et comparer plusieurs datasets (à quelles approches ils correspondent)

Tout modèle predictif repose sur un jeu de données. Sans donnees  pas de ML.
En 1978, David Aha, étudiant a University of California Irvine a crée un serveur mettant a disposition des jeux de données pour le machine learning. Ce serveur est maintenant une source incontournable des données pour le machine learning avec plus de 600 jeu de données, repertorié par tache et type de donnée et vous allez l'explorer.

- Allez sur le site https://archive.ics.uci.edu/datasets
- le premier jeux de données est Iris, cliquez sur lavignette vous observez
    - la  tache: classification
    - le nombre d'echantillon: 150
    - le  nombre de variable predictire: attributes: 4
    - la date de donation: 1 juillet 1988
    etc
dans le menuy de gauche utilisez  les filtres p[our trouver les dataset
- data type: tabular
- tasK: on retrouve nos 3 grand groupe : classification, regression et clustering
- cliquez sur l'une des vignettes  et explorez la partie features
- La page montre les performances de certains algo
- et les papiers qui utilisent ce dataset (il y en a bien plus )

A votre tour.
- trouvez un dataset
- regardez ses variables
- ses caracteristiques ...

Il y a d'autres sources de dataset, nous reviendrons dessus au chapitre ...

Nous avons beaucoup parlé de modeles dans ce chapitre.
Nous allons maintenant preciser ce terme fourre tout dans  le contexte du ML.
