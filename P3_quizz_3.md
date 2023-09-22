Dataset

https://data.ademe.fr/datasets/dpe-v2-tertiaire-2


------------------------------------------------
Q1) A quoi sert l'approche data-centric par rapport à une approche model-centric?
------------------------------------------------

- choisir et optimiser un modèle est un problème résolu. les gains de performance potentiels résidant dans la qualité des données
- les modèles sont centrés sur les données, il n'y a pas de différence fondamentale entre les 2 approches
- l'approche data-centric consiste à établir une suite de règles de classification sans faire intervenir de modèle

feedback:

la réponse 1 est la bonne question de cours

------------------------------------------------
Q2) pourquoi faut il toujours donner une valeur au paramètre random_state
------------------------------------------------

- pour pouvoir reproduire vos expériences et obtenir les mêmes résultats à chaque fois
- pour indiquer l'ordre donnees avant le split entre sous ensemble de test et d'entraînement
- pour que la création du jeu de données soit plus rapide lorsque l'on utilise make_blobs ou make_moons

feedback:

la réponse 1 est la bonne question de cours

------------------------------------------------
Q3) Qu'est-ce que le biais de préjugé dans le contexte des modèles de machine learning ?
------------------------------------------------

- Cela désigne les préjugés ou les stéréotypes présents dans les données d'entraînement d'un modèle.
- C'est une mesure de la précision d'un modèle de régression.
- Cela signifie que le modèle préfère certaines catégories de données aux autres.

Réponse 1: Le biais de préjugé dans les modèles de machine learning fait référence à la présence de discriminations injustes dans les résultats du modèle, généralement causées par des biais systémiques dans les données d'entraînement. Atténuer ce biais est essentiel pour garantir des décisions plus équitables et non discriminatoires lorsque ces modèles sont utilisés dans des applications du monde réel.



---
Nous allons travailler maintenant sur le dataset de l'Ademe, sur le dataset Le Diagnostic de Performance Energétique (DPE).
Le site de l'ADEME, Agence de l'environnement et de la maîtrise de l'énergie  offre la possibilité d'extraire 10000 échantillons du jeu de données entier.

https://data.ademe.fr/datasets/dpe-v2-tertiaire-2


Le Diagnostic de Performance Énergétique (DPE) renseigne sur la performance énergétique et environnementale d’un logement ou d’un bâtiment, en évaluant sa consommation d’énergie et son impact en matière d’émissions de gaz à effet de serre.

Vous trouverez une extraction du dataset sur le github du cours
https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/ADEME_dpe-v2-tertiaire-2.csv

Ce dataset comporte 64 variables différentes. Nous allons nous concentrer sur les colonnes suivantes

Les établissements recevant du public (ERP) sont des bâtiments, locaux et enceintes dans lesquels des personnes extérieures sont admises. Par exemple, une école, un commerce, un parc d'attractions sont des ERP.
Les ERP sont classés en 5 catégories en fonction de leur capacité d'accueil. Les salariés sont comptés avec le public admis dans l'établissement sauf pour la 5e catégorie.

------------------------------------------------
Q4) données manquantes
------------------------------------------------


le dataset fait 10000 échantillons en utilisant par exemple df.info()
sur les 64 colonnes,

Vous pouvez obtenir la liste des  colonnes avec  df.columns et le nombre d'échantillons nulles avec
df[df[nom_colonne].isna()].shape

combien de variables sont réellement exploitables, cad ont au  moins 5000 non nulles

- 42
- 65
- 10

solution

colonnes = cols = [col for col in df.columns if df[df[col].isna()].shape[0] <= 5000 ]

retourne la liste des colonnes ayant plus de 5000 échantillons non nul

cette liste comprend  42 éléments


------------------------------------------------
Q5) comment gérer les données manquantes pour une des variables en particulier
------------------------------------------------

La variable "Emission_GES_kgCO2/m²/an" a 4392 échantillons de valeur manquante. soit presque

Quelle serait la meilleure stratégie pour remplacer ces valeurs nulles par des valeurs qui font sens dans le cadre de l'entraînement d'un modèle de ML

- réaliser une régression linéaire a partir des variables les plus corrélés avec cette variable: Surface, période de construction et secteur d'activité
- prendre la moyenne globale pour cette variable sur tout le dataset
- mettre -1 a la place des valeurs manquantes

Bien que les 3 réponses soient en effet des stratégies de remplacement des valeurs  manquantes
La meilleure stratégie serait d'estimer la valeur manquante à partir de données corrélées.

prendre la moyenne globale modifierait fortement la distribution de la variable et aurait un impact le modèle

mettre arbitrairement une valeur négative à la place d'une valeur toujours positive aurait aussi un impact néfaste sur la distribution de la variable


------------------------------------------------
Q6) outliers:
------------------------------------------------

df.describe retourne le min, max, déviation standard et les percentile des chaque variable numérique.

Faites df.describe() et comparez la ligne min, max et la ligne mean des colonnes. Quelles variables ont de façon certaine une présence d'outliers ?
N'hesitez pas a regardez a quoi correspond l'échantillon avec  la valeur qui vous semble aberrante pour vérifier si c'est bien le cas

par exemple pour une valeur de l'année de construction de 1300 vous trouverez

df[df['Année_construction'] == 1300]

l'adresse Lauzasses de Casteljau (Casteljau) 07460 Berrias-et-Casteljau
et de secteur d'activité N : Restaurants et débits de boisson
et en  regardant sur google maps il pourrait bien dater de  l'an 1300!


- [T] Nombre_occupant (max)
- [T] Conso_kWhep/m²/an
- [F] Surface_utile
- [F] Version_DPE
- Année_construction (min)

Le nombre d'occupant maximum est de 292841658.00 soit 292,8 Millions de personnes!
La Conso_kWhep/m²/an a un minimum negatif.

La surface utile de Surface_utile 116912.00 m2 correspond à un centre commercial: Centre commercial Belle Epine.
Donc possiblement très grand.



------------------------------------------------
Outliers
------------------------------------------------

Regardons maintenant les outliers sur la variable 'Emission_GES_kgCO2/m²/an'

Avant cela, pour simplifier les manipulations , supprimons les valeurs absentes

col = 'Emission_GES_kgCO2/m²/an'
df = df[~df[col].isna()].copy()
df.reset_index(inplace = True, drop = True)
Il reste 5608 échantillons

------------------------------------------------
Q7) Z score sur 'Emission_GES_kgCO2/m²/an'
------------------------------------------------

calculons le z-score pour chaque valeur de cette variable
from scipy import stats
df['z_emissions'] = stats.zscore(df[col])

Quelle affirmation est vrai (2 réponses possibles)

- ne garder que les échantillons avec un z_score de max 2, supprime 37 échantillons.
- ne garder que les échantillons avec un z_score de max 3, supprime 11 échantillons.

- ne garder que les échantillons avec un z_score de max 2, supprime 137 échantillons.
- ne garder que les échantillons avec un z_score de max 2, supprime 37 échantillons.

df[df.z_emissions > 2].shape
37
df[df.z_emissions > 3].shape
11

------------------------------------------------
Q8)  détection des outliers avec l'IQR
------------------------------------------------
Calculez l'IQR sur la variable 'Emission_GES_kgCO2/m²/an' avec
col = 'Emission_GES_kgCO2/m²/an'
iqr = np.quantile(df[col], q=[0.25, 0.75])

En reprenant la définition de limite haute et basse  du cours

que pouvez vous dire

- la limite basse est négative ce qui est absurde
- ne garder que les échantillons supérieurs à la limite haute supprimerait 488 échantillons.
- la limite haute est 24.5
- la limite haute IQR est égale à moyenne + 2 * std

------------------------------------------------
Q9) Binarisation de la variable
------------------------------------------------

Comparez les fonctions de binarisation d'une variable continue pd.cut() et pd.qcut() avec  10 bins pour la variable 'Emission_GES_kgCO2/m²/an'
pd.cut(df[col], 10)
pd.qcut(df[col], 10)

En décomptant le nombre d'éléments par bin avec la fonction value_counts()

par exemple pd.cut(df[col], 10).value_counts()

Que constatez vous ?

- cut ne sert à rien car  la plupart des échantillons sont dans un seul intervalle
- qcut reparti les échantillons en intervalles qui ont a peu pres le meme  nombre d'échantillons

- cut reparti les échantillons en intervalles qui ont a peu pres le meme  nombre d'échantillons
- qcut reparti les échantillons en intervalles d'amplitude sensiblement égales


response  1 est vraie
pd.cut(df[col], 10).value_counts()
Emission_GES_kgCO2/m²/an
(-3.126, 312.59]  	5604
(312.59, 625.18]     	1
(1250.36, 1562.95]   	1
(1562.95, 1875.54]   	1
(2813.31, 3125.9]    	1
(625.18, 937.77]     	0
(937.77, 1250.36]    	0
(1875.54, 2188.13]   	0
(2188.13, 2500.72]   	0
(2500.72, 2813.31]   	0


réponse  2 est aussi vraie
les 2 autres réponses inverse cut et qcut

pd.qcut(df[col], 10).value_counts()
Emission_GES_kgCO2/m²/an
(-0.001, 0.1]  	642
(4.3, 6.0]     	574
(6.0, 8.0]     	567
(40.43, 3125.9]	561
(2.7, 4.3]     	560
(10.7, 15.8]   	560
(15.8, 24.5]   	560
(24.5, 40.43]  	558
(8.0, 10.7]    	528
(0.1, 2.7]     	498


------------------------------------------------
Q10 encodage numérique des  variables catégoriques textuelles
------------------------------------------------

10 Période_construction
24 Secteur_activité
11 Type_énergie_principale_chauffage


regardons maintenant la variable catégorique Secteur_activité  qui prend 24 valeurs différentes

Quelles affirmation sont vraie?  (plusieurs  réponses possibles)

- le one hot encoding remplacerai la variable en question par 23 nouvelles variables
- l'encodage binaire  rajouterai 5 colonnes au dataset
- on peut simplement assigner de façon séquentielle un entier a chaque valeur car  la variable est ordinale
- le one hot encoding remplacerai la variable en question par 24 nouvelles variables


le one hot encoding ajoute N-1 variables pour une variable qui prend N valeurs et non pas n nouvelles variables.

l'encodage binaire crée P nouvelles variables avec P tel que 2^P > N
comme 32 = 2^5 > 24, l'encodage binaire rajouterai 5 nouvelles variables

les valeurs de la  variable ne sont pas ordonnées donc la variable n'est pas ordinale. Il ne fait pas sens d'encoder  la variable avec une suite d'entier
