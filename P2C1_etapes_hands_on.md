P2C1 : Maîtrisez les étapes de construction d’un modèle prédictif


## À vous de jouer ! (quick win)

Fixons les idées sur un exemple simple de régression linéaire.

Voici un dataset avec 237 échantillons comprenant l'âge, le sexe, la taille et le poids d'enfants de 11,5 à 20 ans.
Et voici en quelques lignes comment entraîner un modèle prédictif sur ces données.

Pour débuter, nous allons utiliser tous les échantillons du dataset sans le scinder en 2 (entraînement + test)

A noter que

- l'âge est en mois et non en années, donc de 139 mois a 250 mois soit de 11,5 a 20 ans.
- la variable sexe est binaire: 0 pour les garçons et 1 pour les filles
- la taille en cm varie de ... a ...
- et la variable cible, le poids en Kg est comprise entre ... et ...

Nous allons construire un modèle qui prédit le poids de l'enfant à partir des autres variables: âge, sex et taille.

C'est un modèle simple, difficile de faire plus simple. On cherche les coefficients a,b,c tels que

```
poid = a * age + b * sexe + c * taille
```

Voici un exemple ou on ne considère que les variables d'âge et de taille.

Votre exercice consistera à rajouter la variable binaire sexe.

```
import pandas as pd
df = pd.read_csv()

# les variables prédictives
X = df[['age', 'height']]

# la variable cible, le poids
y = df.weight

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

Reprenons chaque ligne  en commencant par les données

On charge le dataset dans une dataframe pandas

`df = pd.read_csv()`

Par convention on note la matrice des variables predictives `X`,

`X = df[['age', 'height']]`

Ici on ne considere que les 2 variables age et taille

De meme par convention on note le vecteur / array de la variable cible: `y`

`y = df.weight`

Maintenant, etape 2, le modele. Onc hoisit la regression lineaire de scikit-learn

```
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
```

Entrainer le modele consiste a appliquer la methode `fit()` sur le modele  en lui fournissant en entrée
les valeurs des variables predictives  `X` et de la variable cible `y`
```
reg.fit(X, y)
```

A ce stade le modele est entierement entrainé sur le jeux de donnée

Enfin, étape 3,  on regarde la performance du modelepar l'intermediaire du score

```
print(reg.score(X, y))
```

La documentation indique qu'il sagit la du  coefficient de détermination `R^2`


Comme il s'agit d'une regression lineaire, le modele s'exprime

`poids_estimé  = a * age +b * taille + bruit `

et les  coefficients `a` et `b` sont  
`print(reg.coef_)`


Le score est appelé R^2 c'est une mesure des variations de la variable cible expliquées par le modèle.
R^2 va de 0 (mauvais) à 1 (parfait).

Les coefficients ...

Info: par convention de notation on note X, la matrice qui contient les variables prédictrices. dans le code c'est une pandas dataframe. On appelle cette matrice la matrice de design.
La matrice de design représente les données d'entrée, où chaque ligne correspond à une observation et chaque colonne correspond à une caractéristique ou variable d'entrée.
y le vecteur de la variable cible. dans le code, une série de pandas ou un array.

A votre tour, allez sur le notebook colab et completez l'exercice pour toutes les variables prédictrices: age, taille et sexe
Il suffit de transformer ajouter la colonne colonne sexe a la matrice de design:
```
X = df[['age', 'height','sex']]
```

Vous devriez observer un meilleur score que sans la colonne fille / garçon.
Qu'observez vous au niveau des coefficients?

Vous pouvez enfin faire une prédiction si un nouvelle élève arrive en cours d'année:
poids = reg.predict([12, 170, 0])


## Minimiser une fonction de coût
 on rebondit sur "a vous de jouer".
 qu’est ce qui se passe quand on entraine un modèle sur des données avec la fonction fit() ?
 Un peu de recul. introduction du concept de fonction de coût et de la recherche d'un minimum local: visualiser une surface, on cherche le minimum absolu.

Dans le chapitre précédent, nous avons abordé l'exemple de l'algorithme de calcul de la racine de 2, mettant en évidence le concept d'erreur d'estimation. Cette erreur nous permet non seulement d'arrêter l'algorithme en dessous d'un certain seuil, mais aussi de mettre à jour progressivement les coefficients du modèle dans le cas de l'exemple du gradient stochastique.

Dans tous les cas, le but principal d'un algorithme de machine learning est donc de minimiser l'erreur d'estimation. Cependant, on peut calculer cela de beaucoup de façons. Donc au lieu de se limiter à une seule méthode de calcul, on généralise l'idée en considérant n'importe quelle fonction qui sert de mesure. On appelle ces fonctions "fonctions de coût", cost function en anglais.

La fonction de coût est un concept essentiel en ML.
C'est une façon spécifique de mesure de l'écart ou la distance entre les prédictions du modèle et les valeurs réelles de la variable cible.

Quand dans scikit-learn (et dans d'autres librairies de ML) on applique la fonction ```fit()``` sur un modèle, on démarre l'algorithme qui va minimiser cette fonction de coût jusqu'à un atteindre un seuil minimum d'erreur.

Comme il s'agit d'une routine hautement optimisée, la fonction de coût est souvent liée au choix du modèle. Par exemple pour la régression linéaire dans scikit-learn la fonction de coût est la MSE ou Mean Squared Error définie comme suit

MSE = (1/n) * Σ (y_i - ŷ_i)²

où :
 n est le nombre d'échantillons d'entraînement,
 ii sont les vraies valeurs cibles pour chaque échantillon i,
 ŷ_i sont les valeurs prédites par le modèle pour chaque échantillon i


# Définissez la mesure de l'erreur de prédiction
 ⇒ importance de l’erreur de prédiction: minimiser la fonction de coût est pareil que réduire l’erreur entre les valeurs vraies et celles prédites par le modèle.
 C'est en observant ses propres erreurs que le modèle digère les données et s'améliore

Interprétation géométrique: Pour ceux d'entre vous qui ont fait des maths, il y a bien évidemment un rapport entre
la minimisation d'une fonction de coût et sa dérive.
En calculant les dérivées partielles par rapport aux paramètres du modèle, on peut déterminer la direction dans laquelle la fonction de coût diminue le plus rapidement. Cela permet au fil de chaque itération de mettre à jour les paramètres du modèle pour converger vers un minimum de la fonction de coût. C'est ce qui est communément réalisé par des algorithmes d'optimisation tels que la descente de gradient.

https://ruthwik.github.io/img/gradientdescent/gradient3d2.JPG

Prenons un exemple, en régression linéaire, telle que nous l'avons implémenté


## Evaluez la performance d'un modèle
 Comment évaluer l'erreur? plusieurs métriques sont possibles: quadratique, relative, absolue, max etc a choisir en fonction du problème (classification ou régression) et du contexte.

La fonction de coût permet de faire fonctionner l'algorithme d'entraînement du modèle.
Mais pour évaluer sa performance une fois entraîné, nous avons besoin de donner un score de performance au modèle.

Ce score est une mesure de la performance du modèle sur un ensemble d'échantillons qui peut être l'ensemble de test, d'entraînement ou un autre ensemble d'échantillons. Il permet de quantifier à quel point le modèle est capable de fournir des prédictions proches des valeurs réelles de la variable cible.


Conventions: dans la suite on note

- X: le matrice des échantillons (rangées) des variables prédictrices (colonnes).
- y: le vecteur de la variable cible. aussi appelé ground truth
- y^: les prédiction du modèle

Ces fonctions de score dépendent de la tâche ML considérée. Par exemple, un modèle de classification n'est pas évalué de la même manière qu'un modèle de régression.

Dans le contexte d'une régression on a classiquement les 2 scores suivant
- RMSE
- MAPE

Comme la RMSE ou la MAPE est absolue, et non pas un indice entre 0 et 1 par ex, ce qui rend difficile la comparaison entre des modèles dans des contextes différents.
on peut aussi considérer des variantes relatives de ces variables pour faciliter la comparaison entre modèle ou datasets

relative RMSE
relative MAE

et de nombreuses autres variantes.


pas moins de 15 dans la doc sklearn pour la régression
https://scikit-learn.org/stable/modules/classes.html#regression-metrics

16 pour le clustering et 28 pour la classification

Pour donner un exemple, revenons à notre régression sur le dataset des enfants
et calculons ces scores en utilisant les fonctions déjà implémentées de sklearn

qui sont listées ici: https://scikit-learn.org/stable/modules/classes.html#regression-metrics

metrics.mean_squared_error(y_true, y_pred, *)
metrics.mean_absolute_error(y_true, y_pred, *)
metrics.mean_absolute_percentage_error(...)
metrics.d2_absolute_error_score(y_true, ...)
etc


## Appliquez la méthodologie
Similarités et différences des problèmes de régression, classification et clustering: les étapes de modélisation sont similaires mais l'interprétation des résultats va varier en fonction des métriques choisies.

Avant de rentrer plus en détails dans les différences entre régression, classification et clustering, il est utile de souligner la similarité du processus  de modélisation et d'apprentissage qu'ils ont en commun.

En effet, dans ces 3 cas, entraîner  un modèle va consister à enchaîner ces étapes

- definir le probleme: classification, regression, clustering
- choisir le modèle en fonction du problème et des données disponibles
- choisir la métrique d'évaluation, de scoring
- entraîner le modèle
- calculer son score de performance

- loader les data
- transformer / ameliorer les data  (cleaning et feature engineering)
- scinder les data en train / test ou train / test / validation
- entraîner le modèle avec plusieurs configurations de paramètres (si le modèle s'y prête)
- calculer le score de chaque version du modèle.
- et recommencer pour essayer  d'obtenir un meilleur score


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
