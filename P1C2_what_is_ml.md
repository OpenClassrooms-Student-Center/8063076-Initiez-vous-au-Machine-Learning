# sections
## Comprenez l’intérêt du Machine Learning
    __Ce n’est plus une nouvelle le ML est partout. Quelques exemples frappants du ML au quotidien. ML par rapport à la Data Science et à l'IA. Motivation du cours, motivation de l'apprenant et but du projet de machine learning.__

Sans remonter au Turque méchanique du 17e siècle, on peut dire que
L'intelligence artificielle est née en 1957 avec l'invention du Perceptron, première machine de classification automatique d'images.
S'ensuivi divers époques, printemps aussitot suivis d'hivers de l'IA, et ce n'est qu'avec l'explosion dans les 20 dernières années de la puissance de calcul et de stockage des ordinateurs que l'IA a vraiment pris racine dans notre quotidien.

IA est un terme générique qui englobe aussi bien les craintes de fin du monde que l'ingénierie logicielle de pointe.
Dans ce cours, nous parlerons uniquement de machine learning ou apprentissage automatique.
- Le ML est la brique de base de l'IA, la chaîne dans le vélo, le moteur dans la voiture, l'outil ... .
- Le ML a pour but une tache precise: la prédiction.
- Le ML utilise des algorithmes pour developper des modeles predictifs à partir de jeux de données.

Cette fonction prédictive est polymorphe et s'appelle selon les besoins classification, supervision, detection, proposition, prévision ... mais à la base il y a toujours un élément de prédiction.

C'est de cette profusion de modèles prédictifs dont nous profitons. Push de contenu sur les plateformes, surveillance globale, évaluation des risques, IA générative, optimisation des chaînes de production et de ventes, detection des anomalies, prévisions temporelles etc etc

Dans ce cours nous allons travailler sur le machine learning dit classique qui se nourrit de données de petite taille qui tiennent dans un tableu de donnees excel ou google spreadsheet.
On va donc laisser les données de type audio, images, vidéo et l'IA générative au deep learning et  au réseaux de neurones.


Vous avez à votre disposition des jeux de données, des librairies de machine learning exhaustive et bien documentées (scikit-learn), des plateformes de travail (google colab) et ... A vous de jouer!!!

A la base du ML, se trouve le jeu de données ou dataset. Sans data, pas de ML!

## Découvrez les 2 approches en modélisation: stats vs ML;
    Il y a différentes façons d’approcher un problème concrétisé par un jeu de données.
    exemple avec dataset school (age,  poids, taille, G/F…)
    Bien comprendre la différence entre une approche de modélisation statistique et une approche Machine Learning (Pattern recognition et modèle de prédiction ou analyse de  la dynamique entre les variables et modélisation statistique)
    Conséquence directe des 2 approches : utiliser toute la donnée disponible ou en garder une partie pour l'évaluation du modèle.
    Modèle black box ou explicatif; sklearn ou statsmodel

Soit donc un  jeu de données, collecté par des chercheurs, recuperé a partir d'un outil, d'un capteur ...

Vous etes responsable des donnees une plateforme de contenu  en ligne dont le revenu depend des abonnements souscrit et on vous pose 2 questions

- comportement: quel est le profil des utilisaturs qui s'abonnent
- prediction: comment predire si un nouvel utilisateur va s'abonner ou non
dasn les 2  cas la variable cible est clair, c'est l'action de s'abonner. variable binaire

Mais c'est 2 questions correspondent chacune a une approche differente. modelisation statistique ou modelisation predictive.

Dans la question sur le comportemntn, on cherche a savoir comment epxliquer l'acte d'abonnement en fonction des caracteristiques des utilisateur.
On est dans une demarche de cmprehension de la dynamique entre les variables, d'interpretation. est ce que l'age, le revenu, le telephone utilisé influence la probabilite d'abonnement? C'est l'approche statistique qui va chercher a expliciter les relatiosn entre les variables dans un but explicatif, analytique.

Elle s'appuie sur des hypothèses et des modèles statistiques pour analyser les données et tester des hypothèses spécifiques.
tests d'hypothèses et les intervalles de confiance pour évaluer la signification statistique des résultats.



Dans la question de prediction, on ne cherche pas a savoir le pourquoi mais simplement a predire efficacement l'acte d'abonnement. C'est l'approche machine learning. que l'on peut aussi qualifier de boite noire. L'important est la qualité  de la prediction (la performance du modèle) et sa capacité a performer sur des données nouvelles, sa robustesse  ou capacité de generalisation au dela du dataset d'entrainement.

La modélisation statistique elle vise principalement à comprendre les relations entre les variables et à effectuer des inférences statistiques.
l'interprétation des résultats et la compréhension des relations entre les variables.

le ML se concentre sur la prédiction. Elle cherche à développer des modèles qui peuvent généraliser à partir des données existantes pour faire des prédictions sur de nouvelles données.

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

Le but du ML est de developper un modele predictif a partir d'un jeu de données. mais en pratique comment cela se passe t il?

Dans la suite quand on parle de jeux de donnée, pensez a une feuille de type google spreadsheet ou tableur excel. Les variables sont les colonnes, et les echantillons sont les lignes. On distingue la variable cible à prevoir et les autres variables qui permettent potentiellement de predire cette variable cible.

On part d'un jeu de données que l'on suppose exploitable, en gros cela veut dire que les variables ont une relation entre elles. on parle de corrélation, de causalité, ... le dataset n'est pas un regroupement totallement aléatoire de données qui auraient  ete collectees ici  ou la.

Prenons un exemple pour fixer les idées avec un jeux de donnees cpmprenannt l'age, le sexe, la taille et le poids de collégiens. Il semble realiste de supposer que la taille et l'age de l'enfant sont 2 variables fortment liées à son poids.
Par contre la couleur de ses yeux ou les notes obtenues n'ont' a priori que peu de rapport (corrélation) ou d'influence (causalité) sur sa corpulence.

Supposons donc que nous ayons un jeux de donnée qui va nous servir a entrainer  un modele predictoif.
Developper ce modele consiste en ces 2 etapes

- le travail sur la donnee:
Le but de notre modele va etre de faire des bonnes prediction sur des données qui n'ont pas servi lors de son entrainement. principe de generalisation
On va donc couper notre jeux de donneee en 2 partie. une partie pour entrainer le modele, et une partie de test des performance du modele sur  des données non vue. C'est un processus apppelé validation croisée que l'on va repeter plusieurs fois pour s'assurer que le modele performe dasn tous les cas de repartition des ehcantillons dans les jeux d'entrainement et de test.

- le travail sur les parametres du modele pour optimiser sa  performance par rapport le jeu de validation. On va modifier les parametres et observer les performances du model dnas chaque cas.


Note: parfois pour s'assurer que meme apres beaucoup d'optimisation le modele reste capable de performer sur des données nouvelles on coupe le jeu de données en 3 parties. entrainement, test et une partie validation qui va servir a valider la robustesse du modele un fois l'optimisation sur le jeu de test finie.

Mesurer la performance du modele implique aussi d'avoir definie au prealable une metrique d'evaluation. Cette metrique va dependre de la tache a realiser. classification != regression != ranking

## Distinguez l’ approche supervisée de l’approche non-supervisée
    Approches  supervisées ou non-supervisées;
    Définition d'une variable cible;
    On évalue les modèles différemment selon que l'on soit en supervisé ou non-supervisé

Un distinction importante entre 2 types de ML classique / tabulaire

soit on essaie de predire une des variable, soit on pense que les donneees paartiennnt a plusieurs groupes et on va predire a quelle groupe l'echantillon appartient.
approche supervisée: je predis le prix de location de l'appartement en fonction de sa superficie, localisation, etage etc
approche non supervisée, je repartie les appartements en N groupes distinct et le modele associe un appart a un groupe
une autre distinction dans l'approche supervisée vient de la nature de la variable cible que l'on veut predire
si la variable est un chiffre (prix, taille, emission, ...) (on dit qu'elle est continue), alors on parlera de regression
si la variable cible est une categorie, on parle de classification
on parle de classification binaire en face d'un choix binaire: 0 ou 1 , survivra ou non, achete ou non
de classification multi class qund on a plus de 2 categories: on distngue alors la natre ordonnée ou non d'un ensemble: jeune, adulte, vieux ou
peu moyen beaucoup => ordinal
d'une classification multiclass non ordonnée: chat, chien, mouton


Imaginez que vous ayez un ensemble de photos d'animaux et que vous souhaitiez développer un modèle pour les classifier en fonction des différentes espèces.

Dans le machine learning supervisé, vous auriez besoin d'un ensemble de données étiquetées où chaque photo est associée à l'espèce de l'animal correspondant. Par exemple, vous auriez des photos d'éléphants, de lions, de girafes, etc., avec les étiquettes correspondantes. À partir de cet ensemble de données étiquetées, vous entraînez un modèle qui apprend à reconnaître les caractéristiques communes de chaque espèce. Ensuite, vous pouvez utiliser ce modèle pour prédire l'espèce d'un nouvel animal sur la base de ses caractéristiques. C'est un processus supervisé car le modèle apprend à partir d'exemples étiquetés fournis par un humain.

En revanche, dans le machine learning non supervisé, vous n'avez pas d'étiquettes pour les photos. Vous avez simplement un ensemble de données contenant des photos d'animaux, mais sans information sur les espèces spécifiques. Dans ce cas, vous pouvez utiliser des techniques de clustering, telles que le clustering k-means, pour regrouper automatiquement les photos en fonction de leurs caractéristiques similaires. Le modèle recherche des similitudes entre les photos et les regroupe en clusters sans aucune connaissance préalable sur les espèces. Vous pouvez ensuite analyser les clusters formés pour obtenir des informations sur les différentes catégories d'animaux qui pourraient être présentes dans les données.

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
