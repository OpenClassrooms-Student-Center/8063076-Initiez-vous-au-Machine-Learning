# Quiz P2

10h du matin, -30 degrés et un vent à 140km/h sur l'île Biscoe dans l'archipel Palmer en Antarctique. Vous n'avez pas vu le soleil depuis 3 mois. Bien camouflé.e dans la neige, vous observez à la jumelle un groupe de manchots tout en notant rapidement leurs mensurations sur votre tablette.

Vous faites partie de l'équipe du Pr Kristen Gorman https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php
de la station Palmer https://pallter.marine.rutgers.edu/ et vous adorez les manchots!

Votre travail de recherche porte sur

"l'impact de la variabilité de la glace de mer hivernale sur la recherche de nourriture pré-nuptiale des mâles et des femelles manchots"

De retour à la station, vous chargez le dataset que vous avez compilé avec vos collègues et vous vous mettez au travail.

Note sur la traduction penguins => manchots: en francais, les pingouins vivent dans l'hémisphère nord et ils peuvent voler ! Quant aux manchots, ils ne peuvent pas voler et ils vivent dans l'hémisphère sud.

## Le dataset

insert art

Artwork by @allison_horst

Le dataset a bien été compilé sur le terrain par l'équipe du Dr Gorman. Il est analysé par Allison Horst en détail (avec R) sur https://allisonhorst.github.io/palmerpenguins/

Il est composé de 344 échantillons

des variables categoriques

- species, 3 familles de manchots: Adelie (152), Gentoo (124), Chinstrap (68)
- island, sur 3 iles: Biscoe (168) , Dream (124), Torgersen (52)
- sex: 168 mâles et 165 femelles
- year: collectés entre 2007 et 2009

Des variables numériques

- bill_length_mm: la longueur du bec
- bill_depth_mm: la profondeur du bec
- flipper_length_mm: la longueur des nageoires
- body_mass_g: le poids en gramme

insert art bec

Pour charger le dataset en python

pip install palmerpenguins
from palmerpenguins import load_penguins
data = load_penguins()

Le dataset est aussi disponible sur le github du cours
filename = "https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/palmer_penguins.csv"
data = pd.read_csv(filename)

Dans la suite nous utilisons cette version du dataset qui a été mélangée et dont les échantillons avec des valeurs manquantes ont été expurgés

Les données sont ordonnées par espèce et par island. Nous allons mélanger le dataset avec

data = data.sample(frac = 1, random_state = 808)

enlever les échantillons avec des valeurs manquantes. Il reste 333 manchots.
data.dropna(inplace = True)

et reset l'index de la Dataframe
data.reset_index(inplace = True, drop = True)

# Partie 1 régression linéaire

Vous allez explorer le dataset en utilisant une régression linéaire. Vous cherchez à comprendre la relation entre les mensurations des animaux (bill_length etc ) et leur poids

Q1) Dans un premier temps vous réalisez 3 régressions linéaires univariées

body_mass_g ~ bill_length_mm
body_mass_g ~ bill_depth_mm
body_mass_g ~ flipper_length_mm

le code pour la première régression sera donc

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
X = data['bill_length_mm'].values.reshape(-1, 1)
y = data['body_mass_g']

reg.fit(X, y)


En regardant le score donnée par le modèle soit le R^2, reg.score(), quelle est la mensuration la plus prédictive du poids des animaux

y = data['body_mass_g']
reg = LinearRegression()
for col in ['bill_length_mm','bill_depth_mm','flipper_length_mm']:
	X = data[col].values.reshape(-1, 1)
	reg.fit(X, y)
	print("\n",col, reg.score(X, y))

	y_pred = reg.predict(X)
	print(f"RMSE: {mean_squared_error(y, y_pred)}")
	print(f"MAPE: {mean_absolute_percentage_error(y, y_pred)}")


bill_depth_mm
bill_length_mm and bill_depth_mm
flipper_length_mm
species and island

Sol:

le code donne un R^2
- bill_length_mm 0.3474526112888374
- bill_depth_mm 0.22279878346945325
- flipper_length_mm 0.7620921573403914

Comme R^2 mesure le pouvoir de prédiction des variables de la régression sur la variable cible, une valeur élevée indique une meilleure régression. C'est donc la longueur des nageoires flipper_length_mm qui est de loin la plus prédictive du poids de la bête.


Q2)
Considérons maintenant la régression du poids sur toutes les variables numériques

body_mass_g ~ bill_length_mm + bill_depth_mm + flipper_length_mm

et faisons en sorte de standardiser les variables prédictrices avec le MinMax Scaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm']])

Entraînez la regression body_mass_g ~ bill_length_mm + bill_depth_mm + flipper_length_mm

et calculez le RMSE et le MAPE (il vous faut d'abord obtenir les predictions du modele avec la fonction reg.predict(...) )

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
---
y = data['body_mass_g']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm']])
reg = LinearRegression()
reg.fit(X, y)


y_pred = reg.predict(X)
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
print(f"RMSE: {mean_squared_error(y, y_pred)}")
print(f"MAPE: {mean_absolute_percentage_error(y, y_pred)}")

-- question

- [T] la MAPE est petite, le modèle est bon
- [T] La MAPE correspond à une erreur relative. Dans notre cas 7,7%
- [F] la RMSE est très grande, le modèle est excellent
- [F] la RMSE est trop grande, le modèle est mauvais



Q3) par espèce

On va entraîner la même régression mais cette fois en séparant les espèces
on aura donc 3 modèles différents entraînés sur des données différentes

scaler = MinMaxScaler()
reg = LinearRegression()
for espece in ['Adelie', 'Gentoo', 'Chinstrap']:
	df = data[data.species == espece].copy()
	y = df['body_mass_g']
	X = scaler.fit_transform(df[['bill_length_mm','bill_depth_mm','flipper_length_mm']])
	reg.fit(X, y)
	print("--\n",espece, reg.score(X, y))

	y_pred = reg.predict(X)
	print(f"RMSE: {mean_squared_error(y, y_pred)}")
	print(f"MAPE: {mean_absolute_percentage_error(y, y_pred)}")


--
 Adelie 0.5064249444393831
RMSE: 103103.77927187756
MAPE: 0.0688582880470465
--
 Gentoo 0.6280736371428204
RMSE: 92745.44334404985
MAPE: 0.04719861757681761
--
 Chinstrap 0.5038143982587502
RMSE: 72215.44695584572
MAPE: 0.05660976676978926


- Adélie et Chinstrap ont le même R^2, la régression marche aussi bien pour ces 2 espèces
- Adélie et Chinstrap ont des RMSE différentes, la régression ne marche pas aussi bien pour ces 2 espèces
- Gentoo a le plus petit MAPE, la régression est plus pertinente pour cet espece

Sol 1 et 3
R^2 est une mesure absolu donc si 2 modèle ont le même R^2 ils sont aussi performant l'un que l'autre
RMSE est une mesure relative qui va dépendre de la valeur de la variable cible. 2 RMSE différentes pour 2 set de données et donc 2 modèles différents ne permet pas de conclure
MAPE est une mesure relative mais indépendant de l'amplitude de la variable cible donc une MAPE plus petite veut bien dire un modèle meilleur




Q4) Entraînement et validation

On va maintenant scinder le dataset en 2 sous ensembles: test et entraînement
la fonction train_test_split permet de régler la taille de l'ensemble de test test_size
et de fixer l'aspect aléatoire du split avec le paramètre random_state

Nous allons regarder l'influence du paramètre random_state sur le R^2 de la régression

On reprend donc nos données:

scaler = MinMaxScaler()
y = data['body_mass_g']
X = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm']])
reg = LinearRegression()

et on fait varier le random_state de 0 à 19: np.arange(20)

Le code donne

import seaborn as sns
for test_size in [0.2, 0.5, 0.8, 0.9]:

iqrs = []
for test_size in np.arange(0.1,1,0.1):
    score = []
    for random_state in np.arange(200):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        reg.fit(X_train, y_train)
        score.append(reg.score(X_test, y_test))
    # calcul du IQR
    iqr = np.quantile(score, q=[0.25, 0.75])
    print("IQR", test_size, iqr[1] - iqr[0])

    iqrs.append(iqr[1] - iqr[0])

    fig = plt.figure(figsize=(6, 6))
    sns.boxplot(score)
    plt.title(f"test_size {test_size}")
    plt.show()

Maintenant recommencez avec test_size = 0.5 et test_size = 0.8

qu'est ce que vous observez a partir des boxplots?

- [] pour test_size = 0.2, le score ne varie pas tant que ça entre les différents splits. Le choix de random state n'influence que très peu les résultats
- [] pour test_size = 0.2, Le choix de random state influence fortement le score
- [] Le choix de random state influence plus le score que pour test_size = 0.5.


## Régression logistique

Dans cette deuxième partie du quiz vous allez travailler sur la classification avec la régression logistique.

Mais tout d'abord chargez à nouveau le dataset pour être sûr de travailler sur les bonnes données

filename = "https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/palmer_penguins_openclassrooms.csv"
data = pd.read_csv(filename)
data = data.sample(frac = 1, random_state = 808)
data.dropna(inplace = True)
data.reset_index(inplace = True, drop = True)

La variable sex a pour valeur male, female, nous devons d'abord la numériser.

data.loc[data.sex == 'male', 'sex'] = 0
data.loc[data.sex == 'female', 'sex'] = 1
data['sex'] = data.sex.astype('int')
Maintenant le décompte des valeurs de la variable 'sex' doit donner

sex
0	168
1	165
Name: count, dtype: int64

Enfin scindez le dataset en 2 sous ensembles test et train après avoir contraint l'amplitude des variables

scaler = MinMaxScaler()
y = data['sex'].values
X = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm', 'body_mass_g']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


q1) Peut-on prédire le sexe des manchots à partir de leurs mensurations ?

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


Entraînez une régression logistique et regardez la matrice de confusion

La colonne 0 est pour les mâles, la colonne 1 est pour les femelles

clf = LogisticRegression(random_state = 42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

confusion_matrix(y_test, y_pred)
vous pouvez vous aider en regardant directement le nombre de true positive, true negative etc ... avec
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

Q5) Quelle affirmation est vraie?

le modèle est parfait
le modèle a mal classé 9 mâles en femelles
le modèle a correctement classé 32 femelles
le nombre de faux négatifs est de 9

Solution
on obtient
tn, fp, fn, tp
(23, 9, 3, 32)

donc
- réponse 1 fausse: le modèle n'est pas parfait. un modèle parfait aurait des 0 en dehors de la diagonale
- réponse 2 vraie: le nombre de faux positifs est de 9
- réponse 3 vraie: le nombre de vrai positifs est de 32
- réponse 4 fausse: les faux négatifs (mâles classés en femelles) sont de 3

Q6) recall vs precision

Regardez les scores de recall et de précision

recall_score(y_test, y_pred)
precision_score(y_test, y_pred)

En gardant en tête que les positifs sont les 1 et les négatifs les 0
et que notre encodage de la variable cible est
male = 0 ; female = 1,

Quelle affirmation est vrai

Le recall est supérieur à la précision donc
Le modèle est meilleur pour minimiser l'erreur : male => female les faux positifs
Le modèle est meilleur pour maximiser les vrai positifs
Le modèle est meilleur pour minimiser l'erreur : vrai valeur female mais valeur prédite: male (faux négatifs)
Le modèle est meilleur pour maximiser les vrai négatifs


solution 1
c'est dans le cours, le recall est supérieur à la précision
2 et 4 sont fausses car les définition de recall et précision ne permettent pas conclure sur la maximisation des TN et TP
3 est fausse. Ce serait la bonne réponse si la précision est supérieure au recall


Q7) regardons maintenant ce qui se passe quand on fait varier le seuil de classification

Calculez les probabilités d'appartenance à la catégorie female (0 et 1) avec
y_proba = clf.predict_proba(X_test)[:,1]

y_pred_03 = [ 0 if value < 0.3 else 1 for value in y_proba ]
y_pred_07 = [ 0 if value < 0.7 else 1 for value in y_proba ]


Comparez les matrices de confusion pour y_pred_03 et y_pred_07
En gardant en tête que la matrice de confusion est pour scikit-learn

TN | FP
FN | TP

Que peut on affirmer

confusion_matrix(y_test, y_pred_03)
Out[303]:
array([[14, 18],
  	[ 1, 34]])

confusion_matrix(y_test, y_pred_07)
Out[304]:
array([[32, 0],
 	[13, 22]])

[T] Baisser le seuil de classification influence le modèle pour qu'il identifie plus souvent les femelles que les mâles
[F] Augmenter le seuil de classification influence le modèle pour qu'il identifie plus souvent les femelles que les mâles
[F] Baisser le seuil de classification influence le modèle pour qu'il identifie plus souvent les mâles que les femelles
[T] Augmenter le seuil de classification influence le modèle pour qu'il identifie plus souvent les mâles que les femelles

Solution

confusion_matrix(y_test, y_pred_03)
Out[303]:
array([[14, 18],
  	[ 1, 34]])

donc baisser le seuil à 0.3 fait que presque toutes les femelles sont correctement identifiées 34 sur 35. par contre les mâles ont plus d'erreurs: 18 sont identifiés comme femelles et seulement 14 comme mâles


confusion_matrix(y_test, y_pred_07)
Out[304]:
array([[32, 0],
 	[13, 22]])

donc augmenter le seuil fait que tous les mâles sont correctement identifiés 32 sur 32 alors que 13 femelles sur 35 sont identifiés comme mâles

-- prédiction de l'espèce
Q8)
changeons maintenant de variable cible et créons un modele de prediction de l'espèce des manchots

pour cela il nous faut numériser la variable cible de la façon suivante

data.loc[data.species == 'Adelie', 'species'] = 3
data.loc[data.species == 'Gentoo', 'species'] = 2
data.loc[data.species == 'Chinstrap', 'species'] = 1
data['species'] = data.species.astype('int')


ce qui donne la répartition suivante

3  	146
2  	119
1 	68

Construisant la matrice de prédiction et les sous ensembles de train et de test


scaler = MinMaxScaler()
y = data['species'].values

on va comparer 2 modèles,
le premier avec les variables de mensurations sans le poids
X1 = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm']])
X1_train, X1_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
le deuxième auquel on ajoute la variable de poids
X2 = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm', 'body_mass_g']])
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.20, random_state=42)

X2 = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm', 'body_mass_g','sex']])
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.20, random_state=42)

X2 = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm','sex']])
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.20, random_state=42)

enfin entraînons 2 nouveau modèles
clf1 = LogisticRegression(random_state = 42)
clf1.fit(X1_train, y_train)
clf2 = LogisticRegression(random_state = 42)
clf2.fit(X2_train, y_train)

y1_pred = clf1.predict(X1_test)
y2_pred = clf2.predict(X2_test)

Regardez la matrice de confusion des 2 modèles, qu'en concluez vous

- [T] les manchots sont presque entièrement identifiés grâce à leur mesure de bec et de nageoire
- [T] rajouter la variable de poids ne change pas grand chose
- [T] en admettant un seuil d'erreur acceptable, pour identifier les espèces des manchots nous n'avions pas forcément besoin de les peser
- [T] sachant que peser un manchot est compliqué et stressant pour l'animal. Au vu des résultats de classification nous n'avions pas forcément besoin de cette variable pour prédire leur espèce.

- [F] il faut toujours utiliser toutes les variables à disposition
- [F] avec le premier modèle (sans poids) un manchot du sous ensemble de test de l'espèce Gentoo est pris pour un manchot Chinstrap
- [F] avec le deuxième modèle (avec poids) un manchot du sous ensemble de test de l'espèce Gentoo est pris pour un manchot Chinstrap


En effet la seule différence entre les matrices de confusion est que dans le premier modèle sans poid, un manchot de type Chinstrap est confondu avec un manchot Adelie.

Donc les réponses ... sont vraies. nous n'avions pas forcément besoin du poids.
- il ne faut sûrement pas toujours utiliser toutes les variables à disposition. Il pourrait y avoir des effets de fuite d'information ou de fortes corrélation ou d'impact nulle
- En fait au vue de l'encodage, c'est un Chinstrap qui est confondu avec un Adélie et non un Gentoo avec un Chinstrap
- le deuxième modèle est parfait, pas de confusion entre les espèces dans les prédictions sur le sous ensemble de test

## Partitionnement

Nous allons maintenant voir si le kmeans donne un partitionnement exacte des espèces des manchots

Q9) supposons que l'on ne sache pas qu'il y a 3 espèces

Utilisez le score de silhouette pour déterminer le nombre de partition optimum des manchots

sur tout le dataset
en utilisant les variables numériques

X = data[['bill_length_mm','bill_depth_mm','flipper_length_mm', 'body_mass_g','sex']]

from sklearn.cluster import KMeans
km = KMeans( n_clusters=3, random_state = 808, n_init = 10)
km.fit(X)
y_pred = km.labels_
data['labels'] = km.labels_

data[['species', 'labels', 'island']].groupby(by = ['species', 'labels']).count().reset_index().rename(columns = {'island': 'count_'})

donne le tableau suivant qui indique par espèce et par label, le nombre de manchot
	species labels count_
0  	1  	0 	48
1  	1  	2 	20
2  	2  	1 	74
3  	2  	2 	45
4  	3  	0 	102
5  	3  	2 	44

A partir de ce tableau, que pouvez vous conclure

- [F] le partitionnement des manchots recoupe parfaitement la classification en espèce
- [T] il n'est pas possible d'associer une espece particuliere a une des partitions du kmeans
- [F] Adelie (3) correspond pour la plupart des échantillons au groupe 0
- [T] Gentoo (2) correspond pour la plupart des échantillons au groupe 1

Si le partitionnement recoupent parfaitement les espèces nous aurions un count_ plus important par groupe et espèce or ce n'est pas le cas
En particulier, le label 0 est a la fois principale pour l'espece 3 (Adelie) et l'espece 1 (Chinstrap) on ne peut donc pas associer la bonne espèce au label 0
Par contre l'espèce 2 (Gentoo) est principalement associée au groupe 1 qui lui n'est pas considéré pour une autre espèce.


Q10) combien de groupe

utilisez maintenant le score de silhouette pour déterminer le nombre optimal de partitions

from sklearn.metrics import silhouette_score

scores = []
for n in range(2, 11, 1):
	km = KMeans( n_clusters=n, random_state = 808, n_init = 10)
	km.fit(X)
	labels_ = km.predict(X)
	scores.append(silhouette_score(X,labels_ ))

plt.plot(range(2, 11, 1), scores)

Que pouvez vous conclure

- le nombre optimal de clusters est bien de 3
- [T] le nombre optimal de clusters est de 6
- le nombre optimal de clusters est de 9
- Il n'y a pas de nombre optimal de clusters
