## P2C3 : Classifiez avec la régression logistique
  Résolvez un problème de classification
  Questions abordées par la classification: variable catégorielle: plantes et animaux, couleurs, pathologies, intrusion, pannes, …
  Différents types de classification: binaire, multi class, ordonnées etc …


La classification c'est comme la régression, une technique de modélisation supervisée, (on connaît la variable de vérité)
A la différence de la régression, dans la classification la variable cible est une catégorie.

On distingue plusieurs sous cas de classification suivant les valeurs prises par la variable cible

- le cas le plus courant, la classification binaire ou la variable cible prend 2 valeurs exclusives:
oui / non ou 0/1 ou vrai faux ou meurt survit ou triche / triche pas ou marche / marche pas etc etc etc

- le cas multiclasse ou la variable cible peut prendre plus de 2 valeurs exclusives
dans le cas multiclass on distingue le cas où les valeurs sont ordonnées: 1,2,3 ou mauvais, moyen, bon on parle de classification ordinale
et le cas ou les catégories ne sont pas ordonnées: par exemple les couleurs, les nationalités, les animaux etc etc

Enfin, il y a aussi le cas où les catégories ne sont pas exclusives, un échantillon peut appartenir a plusieurs catégories. On parle de classification multi label. Par exemple un film peut appartenir à plusieurs genres à la fois, un plat a plusieurs Catégories Alimentaires à la fois etc etc ou un animal peut appartenir a plusieurs catégories taxonomiques par ex un tigre Mammal, Felidae, Carnivore

Dans la suite nous allons travailler sur la classification binaire et avec 3 classes.

Le principe général de modélisation est similaire, ce qui va surtout changer par rapport à la régression ce sont les métriques de score du modèle et leur interprétation.
On va se concentrer sur l'équivalent de la régression linéaire pour la classification, une modélisation appelée régression logistique.

Mais tout d'abord parlons de l'éléphant dans la pièce, pourquoi parle t-on de régression logistique pour la classification??
C'est pour le moins confusant n'est il pas?


## Comprenez la régression logistique
  la régression logistique : l'équivalent de la régression linéaire mais pour la classification
  encart: passer de la régression à la classification avec la fonction logit. c’est pourquoi on appelle ça la régression logistique et non pas la classification linéaire.

L'idee est simple.
Pour faire de la classification pourquoi ne pas partir de la régression linéaire que nous avons vu
et de trouver un moyen pour que l'on puisse interpréter la prédiction de la régression linéaire, comme une des catégories de la variable cible.

prenons le cas d'une régression à un prédicteur et 2 catégories a prédire: 0 et 1
y = ax + b
Si on pouvait trouver une fonction f telle que
catégorie predite =f(ax+b)
alors on pourrait simplement faire une régression linéaire, transformer le résultat et obtenir une prédiction de la catégorie

et bien cette fonction existe et elle s'appelle .... wait for it ...la fonction logistique !!
d'où le terme de régression logistique pour modéliser une classification

Plus précisément, voilà comment marche une régression logistique

- on fait une régression linéaire
  - y = ax + b
- on transforme y une variable continue en variable dans l'intervalle [0,1]: logit(y)
- on interprète la valeur de logit(y) comme étant un pourcentage, aka la probabilité que y soit dans une catégorie ou l'autre

si logit(y) < 0.5 => prédiction catégorie 0
si logit(y) >= 0.5 => prédiction catégorie 1

Et Pour ceux que ca chatouille vraiment la fonction logit est

f(y) = 1 / (1 + e^(-y))

quelque soit la valeur de y entre -inf et +inf, logit(y) sera comprise entre 0 et 1

donc la régression logistique d'une variable binaire dans le cas de N prédicteurs (x_1,..., x_N) consiste à trouver les coefficients c_1, ... c_N
tels que

y =logit( c_1 x_1 + ... c_N x_N + epsilon )


chatGPT: Un nombre entre 0 et 1 peut être interprété comme une probabilité parce qu'il mesure la possibilité d'un événement se produisant. 0 signifie impossible, 1 signifie certain, et les valeurs entre 0 et 1 indiquent les chances relatives de l'événement.

merci chatGPT :)

On peut dans certains cas modifier le seuil d'appartenance à une catégorie et au lieu de prendre 0.5 (50%) considérer comme seuil n'importe quelle valeur entre 0 et 1.

Prenons un exemple avec le modèle de sklearn LogisticRegression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

un des dataset classique de classification est le dataset du cancer du sein disponible sur UCI
et directement dans sklearn
569 échantillons, une variable cible binaire : malignant ou begnin

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=808, max_iter = 10).fit(X, y)
clf.score(X, y)

Avant d'éplucher le score

regardons un échantillon précisément et sa prédiction

x = [X[8, :]]
clf.predict(x)
on obtient 0: begnin

un autre
x = [X[13, :]]
clf.predict(x)
on obtient 1: malignant

Mais on peut aussi obtenir la probabilité de prédiction de chaque échantillon avec la fonction predict proba
clf.predict_proba([X[13, :]])
array([[0.1193025, 0.8806975]])
donc 88% d'appartenir à la classe 1

clf.predict_proba([X[13, :]])
array([[0.69722956, 0.30277044]])
soit 69% d'appartenir à la classe 0
donc le modèle est moins sûr pour l'échantillon 8 que pour l'échantillon 13

On peut alors tracer l'histogram des probas des predictions du modele
y_hat_proba = clf.predict_proba(X)

import seaborn as sns
sns.histplot(y_hat_proba[:,1])

On voit que pour la plupart des échantillons le modèle est assez sûre de sa prédiction
les échantillons ont une proba soit proche de 0 ou de 1
avec quelques échantillons reparti au milieu

comparons ce graphe avec les prédictions d'un mauvais modèle
(j'ai limité le nombre d'itérations du modèle à 4 au lieu de 100)

on obtient un graphe avec la plupart des pics au milieu. la majorité des échantillons ont une probabilité autour de 0.6 entre 0.55 et 0.65
le modèle n'est pas sûr de lui

bon maintenant rentrons dans le vif du sujet et regardons les métriques de score des modèles de classification



## Evaluez la performance d'un modèle de classification
  métriques associées : accuracy, ROC-AUC et matrice de confusion. rester simple.
  Cas d'un dataset fortement déséquilibré: détection de fraude,
  paradoxe de la accuracy

Dans le cas d'une classification binaire, la façon la plus simple de mesurer l'efficacité du modèle est encore de regarder
le nombre d'échantillons qui ont été bien classé par le model
on appelle ca accuracy ou précision
c'est la fraction
nombre d'échantillons bien classés / nombre d'échantillon au total

Reprenons le modèle de régression logistique que nous venons d'entraîner sur le dataset du cancer du sein
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

on a

y_pred = clf.predict(X)

et une precision de
from sklearn.metrics import accuracy_score
accuracy_score(y, y_hat)
> 0.7715289982425307

77% pas franchement génial

On peut se poser la question de savoir si le modèle est biaisé

par exemple a t il tendance à classer les 0 en 1 ou les 1 en 0 ou est illegalement mauvais sur les 2 catégories?

Pour cela on fait appel à la matrice de confusion
La matrice de confusion est dans le cas binaire un tableau 2x2
avec pour chaque catégorie
 en colonne le nombre de valeur vrai
et en rangé le nombre de valeur prédite
ou le contraire je neme souviens jamais, c'est confusant


sur la diagonale le nombre d'échantillons bien classé
hors diagonale, les échantillons mal classé

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)

array([[193, 19],
    [ 11, 346]])

donc il classe

...

=> ROC_AUC


## Demo
  Screencast de démonstration d'une régression logistique sur un dataset classique avec scikit learn, présentation de la documentation,

## À vous de jouer !
  Application à un dataset de classification classique.
  sur une classification (penguin, Iris), sklearn, logistic regression
  interprétation des résultats en fonction de divers métriques
  probabilité d’appartenance à une catégorie
  détection des échantillons mal classés
  ROC-AUC et matrice de confusion
