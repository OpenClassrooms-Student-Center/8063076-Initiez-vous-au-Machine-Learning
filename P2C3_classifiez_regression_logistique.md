## P2C3 : Classifiez avec la régression logistique

## Résolvez un problème de classification

Dans un problème de classification, la variable cible est une catégorie et non une valeur continue comme dans la regression.

On distingue plusieurs types de classifications suivant les valeurs prises par la variable cible

- la classification binaire: la variable cible prend 2 valeurs exclusives:

oui / non; 0/1; vrai /faux; marche / marche pas etc...

- le cas multiclasse: la variable cible peut prendre plus de 2 valeurs

Et parmi ce cas multiclasse, on distingue ensuite:

 - la classification ordinale où les valeurs sont ordonnées: 1,2,3; mauvais, moyen, bon ...

 - et le cas où les catégories ne sont pas ordonnées: les couleurs, les nationalités, les animaux ...

Enfin, on parle de classification multi label quand un échantillon peut appartenir a plusieurs catégories à la fois. Par exemple, les genres d'un film ou la taxonomie d'un animal: Un tigre est à la fois un mammifère, un félin, et un carnivore

Nous allons surtout travailler sur la classification binaire.

---

## Comprenez la régression logistique

Le principe général de modélisation est similaire à celui de la régression, ce qui va surtout changer ce sont les métriques de score des modèles et leur interprétation.

Mais tout d'abord parlons de l'éléphant dans la pièce, pourquoi parle t-on de régression logistique pour la classification??
C'est pour le moins déroutant!


L'idée est simple. On va adapter la régression linéaire au cas de la classification en interprétant la prédiction de la régression linéaire, comme une des catégories prise par la variable cible.

Prenons le cas d'une régression linéaire à un prédicteur
y = ax + b

ici, y est continue et potentiellement peut prendre toute valeur réelle.

Imaginons une fonction f qui projette y dans l'intervalle [0,1]

z = f(y) = f(ax+b)

On peut interpréter la variable z comme étant la probabilité que la prédiction soit dans une catégorie ou dans l'autre avec un seuil par exemple de 0.5.

si f(y) < 0.5 => catégorie 0
si f(y) >= 0.5 => catégorie 1

chatGPT: Un nombre entre 0 et 1 peut être interprété comme une probabilité parce qu'il mesure la possibilité d'un événement se produisant. 0 signifie impossible, 1 signifie certain, et les valeurs entre 0 et 1 indiquent les chances relatives de l'événement.

merci chatGPT :)


On peut donc simplement faire une régression linéaire, projeter le résultat dans l'intervalle [0,1] et interpréter cette valeur comme la probabilité d'appartenance à une des catégories souhaitée.

Cette fonction de projection de toute valeur réelle dans l'intervalle [0,1] existe et elle s'appelle la fonction logistique

https://fr.wikipedia.org/wiki/Fonction_logistique_(Verhulst)

f(y) = 1 / (1 + e^(-y))

=> graphe

D'où l'appellation régression logistique pour parler de l'équivalent de la régression linéaire mais pour la classification.

Plus précisément, une régression logistique dans cas de classification binaire de prédiction consiste à:

- modéliser une régression linéaire
- projeter la prédiction y variable dans l'intervalle [0,1]: logistique(y)
- interpréter logistic(y) comme probabilité que y soit dans une catégorie ou dans l'autre en fonction d'un seuil t=0.5

si logistic(y) < 0.5 => catégorie 0
si logistic(y) >= 0.5 => catégorie 1


Prenons un exemple avec le modèle de sklearn LogisticRegression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Considerons le dataset du cancer du sein disponible sur UCI http://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic et directement dans sklearn.

Ce dataset a 569 échantillons, 30 prédicteurs et une variable cible binaire : la tumeur est maligne (aïe, 1) ou bénigne (ouf, 0)

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=808).fit(X, y)

Regardons 2 échantillons en particulier et leur prédiction

x = [X[8, :]]
clf.predict(x)
on obtient 0: bénigne

un autre
x = [X[13, :]]
clf.predict(x)
on obtient 1: maligne

On peut aussi obtenir la probabilité de prédiction de chaque échantillon avec la fonction predict_proba() qui donne la paire de probabilité pour 0 et 1.

Note dans le cas binaire: p(0) + p(1) = 1

clf.predict_proba([X[13, :]])
array([[0.1193025, 0.8806975]])

Soit 88% d'appartenir à la classe 1!

clf.predict_proba([X[13, :]])
array([[0.69722956, 0.30277044]])
soit 69,7% d'appartenir à la classe 0!

Le modèle semble moins sûr de sa classification pour l'échantillon 13 que pour l'échantillon 8

Un bon moyen d'analyser les performances d'un modèle de classification est de tracer l'histogram des probabilités des prédictions

y_hat_proba = clf.predict_proba(X)

import seaborn as sns
sns.histplot(y_hat_proba[:,1])

le modèle est assez confiant de ses prédiction, la plupart des prédictions ont une probabilité proche de 0 ou de 1

Pour un mauvais modèle nous aurions par exemple des prédictions moins clairement espacées

Ici, la majorité des prédictions ont une probabilité entre 0.55 et 0.65. Le modèle est moins confiant

-------

## Evaluez la performance d'un modèle de classification
Le plus simple pour mesurer la performance d'une classification est de regarder le nombre d'échantillons qui ont été correctement classés. C'est l'exactitude ou accuracy en anglais.

exactitude = échantillons bien classés / échantillons au total

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

Reprenons le modèle de régression logistique que nous venons d'entraîner, son exactitude est

y_pred = clf.predict(X)

from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred)
> 0.947

94.7%! c'est un bon score pour une simple régression logistique.

Ce modèle est-il biaisé ou objectif?

A-t-il tendance à classer plutôt les 0 en 1 ou les 1 en 0?

Pour voir cela on fait appel à la matrice de confusion, un tableau 2x2 avec en colonne le nombre de valeur vrai et en rangés le nombre de valeur prédite
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)

array([[193, 19],
 [ 11, 346]])

De façon générale, dans le cas binaire, la matrice de confusion s'exprime de la façon suivante

     | categorie predicte
     | pos | neg
cat reelle pos  | TP | FN
cat reelle neg  | FP | TN

Donc
- la diagonale contient le nombre d' échantillons bien classés
- hors diagonale, le nombre des échantillons mal classés

ou
- TP: true positif, vrai positif: échantillons positifs prédits comme positifs
- FP: false positif, faux positif: échantillons négatifs prédits comme positifs de façon erronée
- TN: true negative, vrai négatif: échantillons négatifs prédits comme négatifs
- FN: false negative, faux négatif: échantillons positifs prédit comme négatifs de façon erronée

A partir des ces métriques TP, TN, FP, FN, on définit de nombreuses métriques adaptées à des problématiques et des interprétations spécifiques

Par exemple, le rappel = TP / (TP + FN) (recall en anglais) est adapté quand les conséquences de manquer des instances positives sont graves. Dans un contexte médical, ne pas détecter la maladie (faux négatif), peut avoir des conséquences sérieuses pour le patient.

La precision = TP / (TP + FP)

La précision est adaptée quand on souhaite minimiser les faux positifs. Un email faussement identifié comme étant du spam (faux positif) peut supprimer des informations importantes. Une haute précision garantit que le modèle réduit les risques de fausses alertes.

La page wikipedia sur la matrice de confusion est particulièrement riche en exemples de métriques en tout genre.
https://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
Je vous laisse vous y référer.

Le paradoxe de l'exactitude

Il y a parfois un fort déséquilibre entre les cas négatifs et ceux positifs.
Par exemple dans la détection de pannes de machines, la détection des maladies, ou la fraude.

Dans tous ces cas, le dataset d'entraînement sera déséquilibré.
Prenons par exemple le cas d'un dataset de fraude avec 1% des échantillons correspondant au cas positif ou il n'y a pas de fraude.

un modele de prediction absurde qui prédit qu'il n'y a jamais de fraude aura une exactitude de 99%
Un score excellent mais qui évidemment ne sert à rien.

D'ou l'intérêt d'avoir des métriques de classifications adaptées à chaque type de situation.


Notes
Ces scores sont dépendants du seuil de classification mentionné auparavant.
Par défaut, on considère un seuil de 0.5.
Si la probabilité prédite < 0.5 alors la prédiction est la catégorie 0 sinon 1
Mais on peut considérer d'autre valeurs pour ce seuil

on part des probabilités prédites

y_hat_proba = clf.predict_proba(X)[:,1]
On obtient les catégories relatives pour les 2 seuils:

y_pred_04 = [ 0 if value < 0.4 else 1 for value in y_hat_proba ]
y_pred_06 = [ 0 if value < 0.6 else 1 for value in y_hat_proba ]

on a alors les matrics de confusion suivantes

pour 0.4
confusion_matrix(y, y_pred_04)

pour 0.6
confusion_matrix(y, y_pred_06)

et les recall et précisions récapitulées dans le tableau suivant

from sklearn.metrics import precision_score, recall_score
precision_score(y, y_pred)
precision_score(y, y_pred_04)
precision_score(y, y_pred_06)

recall_score(y, y_pred)
recall_score(y, y_pred_04)
recall_score(y, y_pred_06)

On observe bien que les scores dépendent des seuils considérés

Regardons pour finir une métrique de classification assez généraliste pour être utilisable dans la plupart des cas. Il s'agit de la ROC AUC.

L'idée est de regarder la courbe donnée en traçant le rappel ou TPR (true positive rate) par rapport au FPR (false positive rate) en fonction des seuils de classification.

le TPR est défini par TP / (TP + FN) et le FPR est définie par FP (FP + TN)

La courbe obtenue est appelée Receiver operating characteristic (ROC) ou fonction d’efficacité du récepteur

le graphe suivant montre un bon modèle et un mauvais modèle

=> good_bad_ROC...

la diagonale correspond à un modèle purement aléatoire
Plus la courbe se rapproche du coin en haut à gauche, meilleur est le modèle

La métrique qui nous intéresse est la surface entre la courbe et la diagonale qui est appelée ROC- AUC pour Area under the ROC Curve ou surface sous la courbe ROC.

Dans scikit-learn, il s'agit de la fonction roc_auc_score
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

pour notre modèle, on a une ROC-AUC de

>> 0.99
Excellent!


On peut aussi tracer la courbe ROC

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, y_hat_proba)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)

roc_curve.png





## Demo
Screencast de démonstration d'une régression logistique sur un dataset classique avec scikit learn, présentation de la documentation,


Et si on a plus de categorie pour la variable cible?
par exemple 3

Ca tombe bien, le dataset seminale en classification a 3 categorie.
Il s'agit du dataset Iris, que nous avons deja rencontre sur UCI

Nous avons 3 sortes de plantes et 4 predicteurs: longeur et largeur des petales et ...

Entrainons un regression logistique et regardins lamatrice deconfusion, ROC AUC et laprecision
...








...

=> ROC_AUC


## À vous de jouer !
 Application à un dataset de classification classique.
 sur une classification (penguin, Iris), sklearn, logistic regression
 interprétation des résultats en fonction de divers métriques
 probabilité d’appartenance à une catégorie
 détection des échantillons mal classés
 ROC-AUC et matrice de confusion
