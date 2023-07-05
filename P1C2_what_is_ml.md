# sections
## Comprenez l’intérêt du Machine Learning
    Ce n’est plus une nouvelle le ML est partout. Quelques exemples frappants du ML au quotidien. ML par rapport à la Data Science et à l'IA. Motivation du cours, motivation de l'apprenant et but du projet de machine learning.

ML est une vielle techno, annees 50,
Apres differentes periodes de tattonements, l'IA a un premier printemps en 2000 et depuis l'IA n'a cessé de s'ameliorer.
le ML est la brique de base de l'IA, le moteur dans la voiture, les rouages de l'eolienne et le sujet de ce cours.
IA, ML, Data science ?
l'IA est le domaine generale, la data science, le projet qui va du besoin business a la mise en production, et le ML la methode qui developpe des modeles predictifs.
le ML est le moteur de multiples services et applications que nous utilisons tous les jours. de spotify, netflix, les reseaux sociaux mais aussi la surveullance, la detection de fraude, la santé etc etc  la liste est longue
et depuis ... 2022 le tsunami de l'IA generative (chaatGPT, MidJourney etc ) concrétise une IA du quotidien dans nos vies personnelles et professionnelles.
Nous nageons en pleine science fiction. Exciting!
Dans ce cours nous allons focus sur la brique de base, le moteur de l'IA, le machine learning dit classique.
Le machine learning classique consiste a développer des modeles de predictions a partir de jeux de donnees de taille petite ou faible.
On parle de données tabulaires. de données qui rentre dans un tableu de donnees excel ou google spreadsheet.
On ne parle pas de données audio, image, video ni d'IA generative
Donc a la base du ML est la donnée! la data!

## Découvrez les 2 approches en modélisation: stats vs ML;
    Il y a différentes façons d’approcher un problème concrétisé par un jeu de données.
    exemple avec dataset school (age,  poids, taille, G/F…)
    Bien comprendre la différence entre une approche de modélisation statistique et une approche Machine Learning (Pattern recognition et modèle de prédiction ou analyse de  la dynamique entre les variables et modélisation statistique)
    Conséquence directe des 2 approches : utiliser toute la donnée disponible ou en garder une partie pour l'évaluation du modèle.
    Modèle black box ou explicatif; sklearn ou statsmodel

Soit donc un  jeu de données, provenant d'une equipe, d'un outil ...
Vous etes le data dude et on vous pose 2 questions
- comportement: qui sont les gens qui s'abonnenet le plus et quand
- prediction: on voudrait predire si un user va s'abboner ou non ?

Une de ces questiosn a potentillement une reponse ML, l'autre c'est des stats.

La modélisation statistique vise principalement à comprendre les relations entre les variables et à effectuer des inférences statistiques.
hypothese => tests => validation. on est dans le domaine du A/B testing,etc
Elle s'appuie sur des hypothèses et des modèles statistiques pour analyser les données et tester des hypothèses spécifiques.
 tests d'hypothèses et les intervalles de confiance pour évaluer la signification statistique des résultats.
 l'interprétation des résultats et la compréhension des relations causales entre les variables.

le ML se concentre généralement sur la prédiction et la classification des données. Elle cherche à développer des modèles qui peuvent généraliser à partir des données existantes pour faire des prédictions sur de nouvelles données.
l'interprétation des résultats et la compréhension des relations causales entre les variables.
approche black box

Donc en ML la question est toujours: comment predire
dans un cas on interprete, dans l'autre on prédit
et on interprete si cela est possible
certains modeles de ML se prete plus ou moins a une interpretation a posteriori. c'est aussi un prerequis specifique dans certaines industries.



## Comprenez en quoi consiste le Machine Learning
    Construire un modèle prédictif consiste à l'entraîner sur des données, puis évaluer sa performance et enfin l’optimiser.
    Attentes: le modèle prédictif doit être capable d'être performant sur des données nouvelles, c’est la capacité de généralisation, d'extrapolation  

    concepts: entraînement, évaluation, optimisation, généralisation,

Donc ML = predire
mais comment fais t on
eu final le processus est assez simple. voici un aperçu

On part d'un jeu de données que l'on suppose exploitable
cad il y a du signal, la variable a predire est en partie dictée par les autres variables.
penser superficie du logement et prix de location, profile utilisateur et detection de fraude, poids de la voiture et impact Co2

developper un modele de prediction consiste en ces 2 etapes

- le travail sur la donnee: separer la data en 2: une partie entrainement, et une partie servant a la validation. cdela adresse la demande pricipale  que l'on fait a un modele de ML: sa resilience face a de nouvelles données. Le modele doit etre capable de predire sur des donnees nouvelles qu'il n'a pas deja vu. evidemment on suppose (et ce n'est pas toujours le cas) que les donnees nouvelles ressemble (d'un piouint de vue statistque) aux donnees d'entrainement
- le travail sur les parametres du modele et son optimisation. on entraine un modele avec different parametre pour trouver les aprametres qui offrent la meilleure performance

Evidemment cela pose la question de la performance du modele. quelle metrique, quel critere etc


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

Une constante en machine learning est l'existence d'un jeux de donnee exploitable
Le type du jeux de donnee dicte e queqeu sorte son exploitation
un bon jeu de donnee possede un dictionnaire de donnee

en 1978, ... creation d'UCI, repository de dataset
allez sur UCI, utilisez le filtre et trouvez un jeu de donnee de classification, de regression, de clustering
regardez le dictionnary du jeux de donnees ...
