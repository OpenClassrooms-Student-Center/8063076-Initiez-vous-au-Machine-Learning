P2C1 : Maîtrisez les étapes de construction d’un modèle prédictif


## À vous de jouer ! (quick win)

Fixons les idées sur un exemple simple de régression linéaire.

Voici un dataset avec 237 échantillons comprenant l'âge, le sexe, la taille et le poids d'enfants de 11,5 à 20 ans.
Et voici en quelques lignes comment entraîner un modèle prédictif sur ces données.

https://github.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/blob/master/data/age_vs_poids_vs_taille_vs_sexe.csv

Pour débuter, nous allons utiliser tous les échantillons du dataset sans le scinder en 2 (entraînement + test)

A noter que

- l'âge est en mois et non en années, donc de 139 mois a 250 mois soit de 11,5 a 20 ans.
- la variable sexe est binaire: 0 pour les garçons et 1 pour les filles
- la taille en cm varie de 128.27 cm à 182.88 sm
- et la variable cible, le poids en Kg est comprise entre 22.9 kg  et 77.78 kg

Nous allons construire un modèle qui prédit le poids de l'enfant à partir des autres variables: sexe, âge et taille.

C'est un modèle simple, difficile de faire plus simple.
On cherche les coefficients a,b,c tels que

```
poids = a * sexe + b * age + c * taille +  du bruit
```
où le _bruit_ represente l'information qui n'est pas capturée par le modele linéaire.


Voici un exemple ou on ne considère que les variables de sexe et d'âge.

Votre exercice consistera à rajouter la variable de taille.

voir le notebook colab: https://colab.research.google.com/drive/1YfEEIm3UaQh-AL8V-zqpeIvVpNeHWgMA#scrollTo=AiVXisuqQ9lg

```
import pandas as pd
df = pd.read_csv(<le  fichier csv>)

# les variables prédictives
X = df[['age', 'taille']]

# la variable cible, le poids
y = df.poids

# on choisit un modèle de régression linéaire
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# on entraîne ce modèle sur les données avec la méthode fit
reg.fit(X, y)

# et on obtient directement un score.
print(reg.score(X, y))

# ainsi que les coefficients a,b,c de la régression linéaire
print(reg.coef_)
```

Voilà c'est tout!

Reprenons chaque ligne en commençant par les données

On charge le dataset dans une dataframe pandas

`df = pd.read_csv()`

Par convention on note la matrice des variables prédictives `X`,

`X = df[['age', 'height']]`

Ici on ne considère que les 2 variables âge et taille

De même par convention on note le vecteur / array de la variable cible: `y`

`y = df.weight`

Maintenant, étape 2, le modèle. On Choisit la régression linéaire de scikit-learn

```
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
```

Entraîner le modèle consiste à appliquer la méthode `fit()` sur le modèle  en lui fournissant en entrée
les valeurs des variables prédictives  `X` et de la variable cible `y`
```
reg.fit(X, y)
```

A ce stade le modèle est entièrement entraîné sur le jeux de donnée

Enfin, étape 3,  on regarde la performance du modèle par l'intermédiaire du score

```
print(reg.score(X, y))
```

La documentation indique qu'il s'agit la du  coefficient de détermination `R^2`

Comme il s'agit d'une régression linéaire, le modèle s'exprime

`poids estimé  = a * sexe + b * age + du bruit `

et les  coefficients `a` et `b` sont  donnés par `print(reg.coef_)`

```
[-2.06, 0.3 ]
```

Le score, coefficient de détermination, est une mesure des variations de la variable cible expliquées par le modèle.
R^2 va de 0 (mauvais) à 1 (parfait).

Info: par convention de notation on note
- X, la matrice qui contient les variables prédictrices. Dans le code c'est une pandas dataframe. On appelle cette matrice la matrice de design. La matrice de design représente les données d'entrée, où chaque ligne correspond à une observation et chaque colonne correspond à une caractéristique ou variable d'entrée.
- y le vecteur de la variable cible. Dans le code, c'est une série de pandas ou un array.

A votre tour, allez sur le [notebook colab](https://colab.research.google.com/drive/1YfEEIm3UaQh-AL8V-zqpeIvVpNeHWgMA#scrollTo=AiVXisuqQ9lg) et completez l'exercice pour toutes les variables prédictrices: age, taille et sexe. Il vous suffit d'ajouter la colonne _taille_ à la matrice de design:

```
X = df[['sexe','age', 'taille']]
```

Vous devriez observer un meilleur score que sans la colonne taille. Qu'observez vous au niveau des coefficients?

Vous pouvez enfin faire une prédiction si un nouvelle élève arrive en cours d'année:
Par exemple, pour un garçon (0) agé de 150 mois

```
poids = reg.predict(np.array([[0, 150]]))
```

## Minimiser une fonction de coût

Dans le chapitre précédent, nous avons abordé l'exemple de l'algorithme de calcul de la racine de 2, mettant en évidence le concept d'erreur d'estimation. Cette erreur nous permet non seulement d'arrêter l'algorithme quand l'erreur atteint un certain seuil, mais aussi de mettre à jour progressivement les coefficients du modèle dans le cas de l'exemple du gradient stochastique.

Dans tous les cas, le but principal d'un algorithme de machine learning est de minimiser cette erreur d'estimation. Mais on peut estimer cet écart de beaucoup de façon. Donc au lieu de se limiter à une seule méthode de calcul, on généralise l'idée en considérant n'importe quelle fonction qui puisse servir de mesure. On appelle ces fonctions "fonctions de coût", où _cost function_ en anglais.

La fonction de coût est un concept essentiel en ML. C'est une façon spécifique de mesure de l'écart (distance) entre les prédictions du modèle et les valeurs réelles de la variable cible.

Quand dans scikit-learn (et dans d'autres librairies de ML) on applique la fonction ```fit()``` sur un modèle, on démarre l'algorithme qui va minimiser cette fonction de coût jusqu'à un atteindre un seuil minimum d'erreur.

Comme il s'agit d'une routine hautement optimisée, la fonction de coût est souvent liée au choix du modèle. Par exemple pour la régression linéaire dans scikit-learn la fonction de coût est la MSE ou Mean Squared Error définie comme suit

MSE = (1/n) * Σ (y_i - ŷ_i)²

où :

- n est le nombre d'échantillons d'entraînement,
- y_i sont les vraies valeurs cibles pour chaque échantillon i,
- ŷ_i sont les valeurs prédites par le modèle pour chaque échantillon i


# Définissez la mesure de l'erreur de prédiction

Interprétation géométrique: Pour ceux d'entre vous qui ont fait des maths, il y a bien évidemment un rapport entre
la minimisation d'une fonction de coût et sa dérivée. En estimant les dérivées partielles par rapport aux paramètres du modèle, on peut déterminer la direction dans laquelle la fonction de coût diminue le plus rapidement. Cela permet, au fil de chaque itération, de mettre à jour les paramètres du modèle pour converger vers un minimum de cette fonction de coût. C'est ce qui est communément réalisé par des algorithmes d'optimisation tels que la descente de gradient.

https://ruthwik.github.io/img/gradientdescent/gradient3d2.JPG

## Evaluez la performance d'un modèle

La fonction de coût permet de faire fonctionner l'algorithme d'entraînement du modèle.
Mais pour évaluer sa performance une fois entraîné, nous avons besoin de calculer un score de performance au modèle.

Ce score est une mesure de la performance du modèle sur un ensemble d'échantillons qui peut être l'ensemble de test, d'entraînement ou un autre ensemble d'échantillons. Il permet de quantifier directement la qualité des predictions du modele.


Conventions: dans la suite on note

- X: le matrice des échantillons (rangées) des variables prédictrices (colonnes).
- y: le vecteur de la variable cible. aussi appelé ground truth
- y^: les prédiction du modèle

Ces fonctions de score dépendent de la tâche ML considérée. Par exemple, un modèle de classification ne sera pas évalué de la même manière qu'un modèle de régression.

Dans le contexte d'une régression on a classiquement les 2 scores suivant

- RMSE =
- MAPE

Comme la RMSE ou la MAPE est absolue, et non pas un indice entre 0 et 1 par ex, ce qui rend difficile la comparaison entre des modèles dans des contextes différents.
On peut donc aussi considérer des variantes relatives de ces metriques:

relative RMSE

Il existe de nombreuses autres variantes de  metriques d'evaluation


- pas moins de 15 dans la doc sklearn pour la régression
https://scikit-learn.org/stable/modules/classes.html#regression-metrics
https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

- 16 pour le clustering
- 28 pour la classification

Pour donner un exemple, revenons à notre régression sur le dataset des enfants
et calculons ces scores en utilisant les fonctions déjà implémentées de sklearn

qui sont listées ici: https://scikit-learn.org/stable/modules/classes.html#regression-metrics

- metrics.mean_squared_error(y_true, y_pred, *)
- metrics.mean_absolute_error(y_true, y_pred, *)
- metrics.mean_absolute_percentage_error(...)
- metrics.d2_absolute_error_score(y_true, ...)
etc


## Appliquez la méthodologie
Avant de rentrer plus en détails dans les différences entre régression, classification et clustering, il est utile de souligner la similarité du processus d'entrainement qu'ils ont en commun.

En effet, dans ces 3 cas, entraîner un modèle va consister à enchaîner ces étapes

- loader les data
- transformer / ameliorer les data  (cleaning et feature engineering)
- scinder les data en train / test ou train / test / validation
- entraîner le modèle avec plusieurs configurations de paramètres (si le modèle s'y prête)
- calculer le score de chaque version du modèle.

et recommencer pour essayer  d'obtenir un meilleur score


D'autre part, le choix de la métrique de performance va influencer l'interprétation de ses performances.
Nous en verrons un exemple dans le cas de la classification ou certaines métriques sont peu adaptées dans certains cas. suspens


## À vous de jouer !
Travail sur scikit learn documentation:
Montrer qu'un modèle existe le plus souvent en version régression et en version classification
Montrer qu'il est possible de choisir parmi plusieurs métriques

Allez  sur les pages des arbres de décision
Il existe 2 versions de ces modèle:une pour la classification une pour la régression
la fonction de coût est le "criterion"
Remarquez qu'il y a plusieurs  choix: Gini,Entropy et Logos

De même, regardez le Ridge modèle.
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

Notez la fonction de coût
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
