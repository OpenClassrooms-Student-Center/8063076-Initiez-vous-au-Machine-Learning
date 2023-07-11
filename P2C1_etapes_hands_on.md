# P2C1 : Maîtrisez les étapes de construction d’un modèle prédictif


## À vous de jouer ! (quick win)
    hands on Presentation de scikit learn. Ouvrir un colab et jouer avec le code qui est donné, le compléter.
    Briques de base sklearn: data, fit, score, predict
    - introduire les conventions de notations: X (design matrix), y (verité), y_hat (prédites). la matrice de design: features * samples: penser à une feuille excel:
    - Exemple de régression linéaire avec sklearn

Fixons les idees sur un exemple simple de regression linéaire.
Voici un dataset avec 237 echantillons comprenannt l'age, le sexe, la  taille et le poids d'enfants de 11,5 a 20 ans.
Voici en quelques  lignes comment entrainer un modele predictif sur ces données.
Pour simplifier nous allons utiliser tout le dataset sans le scinder en 2 (entrainment + test)

A noter  l'age est en mois et non en annees, donc de 139 mois a 250 mois  soit  de 11,5 a 20 ans.

Nous allons construire un modele qui predit le poids de l'enfant a partir des  autres variables: age, sex et taille.

C'est un modele simple, peut etre le plus simple. On cherche les coefficients a,b,c tels que
weight = a *  age + b * sex + c * height


```

import pandas as pd
df = pd.read_csv()
# on va se limiter aux garcons,
df = df[df.sex == 0]
# les variables predictives
X = df[['age', 'height']]
# la variable cible, le poids
y = df.weight
# on choit un modele de  regression lineaire
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# on entraine ce modele sur les données avec la methode fit
reg.fit(X, y)
# et on obtient directement un score.
reg.score(X, y)
# ainsi que les coefficients a,b,c de la regression lineaire
reg.coef_
```

Voila c'est tout

A votre tour, allez sur le notebook colab et completez l'exercice pour  les filles et pour tout le dataset
Por tout le dataset, fille et garcons il faut transformer ajouter la colonne colonne sexe aux variable predictrices:
```
X = df[['age', 'height','sex']]
```


## Minimisez une fonction de coût
    on rebondit sur "a vous de jouer".
    qu’est ce qui se passe quand on entraine un modèle sur des données avec la fonction fit() ?
    Un peu de recul. introduction du concept de fonction de coût et de la recherche d'un minimum local: visualiser une surface, on cherche le minimum absolu.

Dans le chapitre ..,. et l'exemple sur l'algorithme de calcul de la racone de 2, on avait mis en avant la notion d'erreur qui permet non seulement de  decider d'arreter l'ago quand celle ci est petite mais aussi, dans lexemple du gradient stochastique,. de mettre a jour les cofficients du modele petit a petit.



On cherche donc a minimiser l'erreur d'estimation.
On note cela de faonc generale par la minimisation de la fonction
erreur = distance(y, y_hat )  

notations / conventions
On note
- X, la matrice de design, les colonnes qui coirrespondent aux variables predictiuves
- y la variable reelle, la verité, le groundtruth
- y^ l'estimation fournie par le  modele.




## Définissez la mesure de l'erreur de prédiction
importance de l’erreur de prédiction: minimiser la fonction de coût est pareil que réduire l’erreur entre les valeurs vraies et celles prédites par le modèle.
C'est en observant ses propres erreurs que le modèle digère les données et s'améliore

## Evaluez la performance d'un modèle
Comment évaluer l'erreur? plusieurs métriques sont possibles: quadratique, relative, absolue, max etc a choisir en fonction du problème (classification ou régression) et du contexte.

## Appliquez la méthodologie
Similarités et différences des problèmes de régression, classification et clustering: les étapes de modélisation sont similaires mais  l'interprétation des résultats va varier en fonction des métriques choisies.

## À vous de jouer !
Travail sur scikit  learn:
Montrer qu'un modèle existe le plus souvent en version régression et en version classification
Montrer qu'il est possible de choisir parmi plusieurs métriques
utiliser ce moment pour faire manipuler des modèles plus complexes que la régression linéaire / logistique
