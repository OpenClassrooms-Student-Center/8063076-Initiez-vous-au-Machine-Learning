# Contraignez votre modèle
Après une première passe sur les données, on essaie d'améliorer le modèle. modifier les paramètres du modèle de régression linéaire / GLM en jouant sur la régularisation L2. introduction de la régularisation comme réglage de la sensibilité / robustesse du modèle.
# Identifiez les types de sous performances
Le biais: le modèle est simplement mauvais
Overfitting ou variance:; Le modèle est trop fortement sensible aux données de training et donc peu performant sur des données nouvelles.
# Implémentez la validation croisée
Comment choisir les bons paramètres du modèle?
Comment être sûr que les données d'évaluation (test) reflètent les données d'entraînement?
=> la technique de la validation croisée ou comment exploiter le jeu de données pour optimiser les paramètres du modèle.
# Détectez l'overfit
Comment détecter si le modèle overfit les données d'entraînement par le biais de la learning curve: les performances du modèles doivent être sensiblement similaire sur les données d'entraînement et de test.
# Demo
reprise des éléments du chapitre
entraîner un modèle et détecter l'overfit par la learning curve
pallier en régularisant
# À vous de jouer !
Hands on: réduire l'overfit
dataset avec quelques outliers dans le train ou le test; perf du modèle faible sur data de validation;
démonstration de l'apport de la cross validation => on obtient de meilleure performances sur le dataset de validation
choix des paramètres

----
Dans ce chapitre nous allons plonger dans la recherche des meilleurs paramètres d'un modèle dans le but d'accroître ses performances.

On a beaucoup parlé jusqu'à maintenant d'optimisation des modèles sans montrer en quoi cela consiste exactement.
En effet, les modèles de régression linéaire et logistique n'ont pas de paramètres.
Dans une régression linéaire tout dépend des prédicteurs utilisés. Il est donc impossible de jouer sur leurs paramètres pour améliorer leur performance.


Quand est ce que le modèle sous performe?

Nous avons vu tout au début de ce cours, au chapitre 2, partie 1, que pour évaluer les perfs d'un modèle, on scinde le dataset en 2 sous ensembles: train et test et que l'on évalue la performance d'un modèle par son score sur le sous ensemble de test.

Dans la suite on note score(train) et score(test) les scores du modèle sur les données d'entraînement et de test.

On peut aussi calculer le score du modèle sur le sous ensemble d'entrainement. Ce score(train) aura d'ailleurs tendance à être meilleur que le score(test). En effet, entraîner le modèle consiste à lui faire apprendre la dynamique et les relations entre les variables dans les échantillons d'entraînement. On suppose que ces relations sont pareils au sens statistique, sur le sous-ensemble de test aussi bien que sur les données réelles qu'il devra prédire une fois mis en production. C'est l'hypothèse de base de toute le machine learning: la similarité des distributions statistiques des sous-ensemble d'entraînement, de test et de production.

Ces 2 scores sur les données d'entraînement et de test vont nous permettre d'identifier un modèle bon d'un modèle mauvais.

1. Le modèle ne comprends rien aux données

Son score(train) est faible, il n'arrive simplement pas à comprendre les données d'entraînement.
Conséquence directe, les données de test seront aussi incomprise, donc le score(test) sera faible

On parle dans ce cas de biais du modèle.

2. Le modèle est bon et sait généraliser

Les score(train) et score(test) sont satisfaisants. Woohooo! Your job is done! Prochaine étape production!

3. Le modèle ne sait pas généraliser

Le score(train) est bon mais le score(test) est faible.

Le modèle colle trop aux données d'apprentissage et ne sait pas extrapoler aux données non déjà rencontré.

On appelle cela l'overfit, le sur-apprentissage.

Le quatrième cas, bon score(test) mais mauvais score(train) reflète la plupart du temps une anomalie statistique et arrive très rarement.

On peut illustrer ces 3 cas par la figure suivante

img/biais-overfit.png


L'optimisation du modèle par le jeu de ses paramètres consiste à trouver le juste milieu entre le biais et l'overfit.

Nous allons illustrer tout cela ainsi que les remèdes potentiels pour pallier les faiblesses du modèle en biais ou en overfit.

Mais avant, fourbissons nos armes. Il nous faut un dataset et un modèle qui dépendent de paramètres.

Pour le dataset, nous reprendrons les arbres de Paris mais dans une version déjà numérisée et nettoyée.
Le code prend trop de place ici, il est disponible dans le notebook du chapitre.

Nous allons essayer d'entraîner un modèle qui prédit le stade de développement de l'arbre. n
Un cas de classification a 4 catégories que nous avons déjà vu: ['Jeune (arbre)', 'Jeune (arbre)Adulte', 'Adulte', 'Mature']
Dans le dataset, ces categories sont numérisées en 1,2,3 et 4.

En ce qui concerne le modèle, nous allons utiliser un arbre de décision comme classificateur.
En l'occurence, le modele DecisionTreeClassifier de scikit-learn
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

Pourquoi un arbre de décision ? Simplement parce que ce type de modèle se prête parfaitement à la démonstration d'overfitting et de biais. Les modele lineaire de regression sont incapable d'overfitter autrement que sur des jeu de donneees jouets, et le KNN serait peu réaliste dans notre contexte.

Qu'est ce qu'un arbre de décision?

Un arbre de décision est un enchaînement de règles de classification établies automatiquement à partir des variables prédictrices.
On peut représenter un arbre graphiquement comme un arbre renversé. Pour l'instant nous nous contenterons de cette introduction simple.

Voici un exemple simple

./img/decision_tree_example.png
Les décisions de type À des plumes ? sur la figure, sont appelées des noeuds et les liens entre les noeuds, des branches.
Les noeuds créent des bifurcations de branches.

Dans notre cas, les règles de bifurcation se feront à partir de critères statistiques sur les variables.

Par exemple:

si [hauteur_m > 10 & libelle_francais = 'Platane' ]
Alors stade de développement = Mature
Sinon [autre regle]

L'algorithme d'entraînement détermine les critères des différentes règles (variables concernées, seuils, valeurs catégoriques) qui sont les plus efficaces pour accompagner la tâche de classification. L'algorithme répond à un critère prédéfini pour établir la regle de bifurcation. Dans la documentation scikit-learn, c'est le paramètre criterion: The function to measure the quality of a split.

Une caractéristique saute au yeux, c'est le nombre de noeuds de l'arbre autrement dit sa profondeur.

Un arbre peu profond a peu de règles, un arbre profond aura beaucoup de règles.
Pour limiter la profondeur d'un arbre de décision, on parle de pruning en anglais, littéralement: élagage.

img/decision_tree_pruned.png
img/decision_tree-deep.png

Dans scikit-learn le paramètre qui règle le prunind s'appelle max_depth ("The maximum depth of the tree. ").

C'est avec ce paramètre que nous allons maintenant jouer pour montrer le biais et l'overfit du modèle.

# Tout d'abord le biais.

Chargeons les données et importons le classificateur

filename = './../data/paris-arbres-numerical-2023-09-10.csv'
data = pd.read_csv(filename)
X = data[['domanialite', 'arrondissement', 'libelle_francais', 'genre', 'espece', 'circonference_cm', 'hauteur_m']]
y = data.stade_de_developpement.values

Scindons les donnees en train et test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.8, random_state=808)


Nous utilisons le DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

Limitons la profondeur de l'arbre a 3.

clf = DecisionTreeClassifier(
    max_depth = 3,
	random_state = 808
)


et entrainons le modele
clf.fit(X_train, y_train)

Pour le score, on utilise l'AUC

from sklearn.metrics import roc_auc_score
train_auc = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
test_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
print("train",train_auc)
print("test", test_auc)

train 0.89
test 0.89

et la matrice de confusion

from sklearn.metrics import confusion_matrix
y_train_hat = clf.predict(X_train)
y_test_hat = clf.predict(X_test)
print(confusion_matrix(y_test, y_test_hat))

[[ 5570  1402   300     2]
 [ 1060  3847  2750     8]
 [  211  1481 13328   466]
 [    4     7   565   877]]

ainsi que le rapport de classification qui donne plusieurs scores

from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_hat))

precision    recall  f1-score   support

1       0.81      0.77      0.79      7274
2       0.57      0.50      0.53      7665
3       0.79      0.86      0.82     15486
4       0.65      0.60      0.63      1453

accuracy                                0.74     31878
macro avg           0.70      0.68      0.69     31878
weighted avg        0.73      0.74      0.74     31878

On voit que les précision (le ratio de bonne pioche parmi tous les positifs) pour les différentes classes ne sont pas très bonnes surtout pour les catégories 'Jeune (arbre)Adulte', Mature' (resp. 0.57 et 0.67, à peine meilleur qu'un pile ou face aléatoire)
et nous observons un recall de 0.5 pour la catégorie 'Jeune (arbre)' (le modèle n'identifie que la moitié des arbres)

Donc un modèle qui a besoin de tendresse.


Notez cependant que le score sur la partie test est similaire au score sur la partie train
clf.score(X_train, y_train)
0.731
clf.score(X_test, y_test)
0.74

C'est ce que je voulais vous montrer. C'est un bon exemple d'un modèle biaisé.


Quels remèdes pour minimiser le biais du modèle?

Plusieurs voies sont envisageables

- ajouter des données. Sur un dataset trop petit, le modèle n'aura pas assez d'exemples pour assimiler les dynamiques internes. Ajouter des données pourra l'aider. On peut soit collecter plus de données et les ajouter au dataset soit utiliser des techniques d'augmentation de données qui créent des échantillons artificiels et gonflent donc artificiellement le dataset d'entraînement. Voir a ce titre ...
- le fameux feature engineering ou l'on va s'efforcer de transformer ou d'ajouter des variables pour encoder plus d'information exploitable par le modèle.
- modifier les paramètres du modèle pour améliorer sa performance. C'est ce que nous allons faire.

Tout cela dépend évidemment du contexte dans lequel vous travaillez.

# L'overfit

Reprenons maintenant notre arbre de décision mais cette fois sans limiter sa profondeur.
Pour cela on set max_depth = None


clf = DecisionTreeClassifier(
	max_depth = None,
	random_state = 808
)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)

Le modèle est en effet meilleur on passe de 0.73 à 0.81

clf.score(X_test, y_test)
0.81

mais on remarque que sur le sous ensemble d'entraînement on a carrément 0.94!
clf.score(X_train, y_train)
0.935048231511254

Nous sommes bien dans un cas d'overfitting où le modèle colle aux données d'entraînement (score(train) excellent) et ne sais pas reproduire la même performance sur les données de test (score(test) faible).

Donc à un moment, entre maxdepth = 3 et max_depth = infini, les scores(train) et score(test) ont divergé. Il nous faut trouver un juste milieu pour ce paramètre.

Essayons de trouver a quel moment cela est arrivé en faisant croître max_depth et en enregistrant les score sur train et test.

L'auc est plus parlant que l'accuracy pour cette démonstration

scores = []
for depth in np.arange(2, 30, 2):
    clf = DecisionTreeClassifier(
      	max_depth = depth,
      	random_state = 808
    )

    clf.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
    scores.append({
      	'max_depth': depth,
      	'train': clf.score(X_train, y_train),
      	'test': clf.score(X_test, y_test),
    })

scores = pd.DataFrame(scores)

on obtient la figure suivante avec en abscisse max_depth, en ordonné, AUC

figs/p3c4_02_overfit.png

Que voit on ?


Quand on augment max_depth, l'AUC(train) croit jusqu'à presque atteindre un score parfait de 1.
Ce score sur train atteint un plateau: augmenter la profondeur de l'arbre ne sert plus a rien a partir de max_depth = 20

l'AUC sur test par contre croît jusqu'à atteindre un maximum autour de 0.94 pour max_depth = 9, 10
Elle décroît ensuite jusqu'à 0.87 pour rejoindre elle aussi un plateau vers max_depth = 20

On voit bien les 3 cas de comportement du modèle
- A gauche, le modèle sous-performe. (biais)
- A droite il overfit
- Au milieu on obtient la meilleure performance sur le test set.

On a ainsi trouvé la valeur optimale du paramètre max_depth pour ce model et ce dataset donné

Comment remedier a l'overfit en générale ?

Un modèle qui overfit est un modèle trop complexe.  Dans notre contexte la complexité se traduit par la profondeur de l'arbre.
Dans le cas d'une régression, la complexité serait traduit par trop de variable prédictives et pour le K-NN, trop de clusters.

La première stratégie de remédiation est d'augmenter la taille du dataset d'entraînement.
En effet plus il y aura d'échantillon plus la complexité du modèle sera diluée dans la masse d'information
Mais cela n'est pas toujours possible.


Ce qui nous amène au concept de régularisation.

--- a mon avis on pourrait splitter le chapitre a ce niveau la. avec un nouveau chapitre qui comprendrait regularisation et corss validation.

# Regularization d'un modele

La régularisation est une technique qui permet de tempérer les ardeurs d'un modèle.
Nous allons rajouter une contrainte sur le modele pour qu'il ne puisse pas coller aux données d'entraînement.

Revenons à la regression lineaire mais cette fois en considérant un modèle de regression qui inclu cette composante de régularisation. Dans scikit learn, nous avons le modèle Ridge qui est simplement une régression linéaire avec de la régularisation
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge

La documentation de scikit-learn pour le modèle Ridge indique

Minimise la fonction de cout: ||y - Xw||^2_2 + alpha * ||w||^2_2

Nous avons vu que définir la fonction de coût permet de fixer l'objectif d'apprentissage au modèle.
Dans la régression linéaire simple la fonction de coût est
||y - Xw||^2_2, ou y est la variable cible, X la matrice des predicteurs et w représente le vecteur de coefficient de la régression linéaire. La fonction de coût est dans ce cas la norme quadratique de l'erreur d'estimation.

Pour Ridge, on ajoute un terme alpha * ||w||^2_2 ou
- ||w||^2_2 joue le rôle d'une contrainte sur les coefficients du vecteur de coefficients de la régression. C'est la regularisation du modele.
- et alpha est le paramètre qui permet de règler l'importance de regularisation

Ce terme ||w||^2_2 empêche les coefficients d'être trop disparate, de diverger. Il lie les coefficients entre eux.
Le modèle ne va donc pas pouvoir accommoder les variations les plus subtiles du dataset d'entraînement et sera forcé de trouver un juste milieu. C'est cette quête du juste milieu qui compense l'overfitting.


Voilà pour un bref aperçu sur la théorie de la régularisation. Ce qu'il faut en retenir  c'est que la  régularisation quand elle est disponible permet dans la plupart des cas de remédier à la overfit.

Note: la regularisation d'un modele peut prendre des formes differentes.

L1 au lieu de L2: Ce terme de régularisation apparaît soit avec la norme quadratique L2 soit la norme de premier degré L1
le terme de régularisation de la fonction de coût sera alors de la forme alpha * |w| ou |w| est la somme de la valeur absolue des coefficients du vecteur w.

Pour les arbres de décision, régulariser le modèle consiste à limiter la profondeur de l'arbre ou à jouer sur d'autres paramètres que nous verrons dans le prochain chapitre.

Dans les réseaux de neurones la régularisation prendre une autre forme que . mais le principe sera le même.


Enfin  il nous faut parler de la validation croisée, mentionnée au chapitre ... partie 1.


# Validation croisée

La repartition arbitraire des ehcantillons en sous ensemble de test et d'entrainement, peut etre problematique si leur caracteristique statistique diverge entre les 2 sous ensemble.

Par un hasard fortuit, certains échantillons pathologiques peuvent se retrouver majoritairement dans un des sous-ensembles.

Imaginez par exemple que dans notre Datassette des arbres de Paris, la plupart des arbres les plus hauts soient dans la partie test.
Le modèle ne verrait que très peu d'arbres haut pendant l'entrainement et et donc serai bien a la peine pour prédire des arbres de cette nature. Son score(test) serait mauvais.

Il faut donc trouver un moyen.

La validation croisée permet de reduire le risque de ces anomalies potentielles dues a une mauvaise répartition des échantillons entre les sous ensembles de test et d'entrainement

Qu'est ce que la validation croisée ?

- nous allons diviser le dataset en K sous-ensembles de taille égales
- à tour de rôle chacun des sous-ensembles jouera le rôle de sous-ensemble de test, les K-1 autres sous-ensembles serviront à entraîner le modèle.
- pour chaque configuration entraînement - test on va calculer le score de performance du modele
- à la fin on choisira le modele qui offre la meilleure moyenne des scores sur toutes les configurations entrainement - test

img/cross-validation-k-fold.png

La validation croisée lors de la sélection des paramètres optimaux pour une dataset et un modèle donné.
Mais elle est aussi utilisée lors de toute expérience (modification des variables, choix de modèle, ...) qui donne lieu à un choix.

Il y a de multiples façons de faire de la validation croisée dans scikit learn. C'est même un peu difficile de s'y retrouver.

Ma méthode préférée pour la selection des parametres est GridSearchCV. Elle offre un juste milieu entre automatisation total et implémentation manuelle complète de la méthode.
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

Utilisons la pour trouver la valeur optimal de max_depth sur notre dataset d'arbres.

from sklearn.model_selection import GridSearchCV
# on defini l'espace des valeurs possibles
parameters = {'max_depth':np.arange(2, 30, 2)}

# on ne precise par le parametre lorsque l'on instancie le modele
model = DecisionTreeClassifier(
	random_state = 808
)
# on passe le model et le dictionnaire des paramtres a GridSearchCV
# cv est le nombre de sous ensemble, de split, de la validationc croisée. On choisit le plus souvent une valeur autour de 5.

clf = GridSearchCV(model, parameters, cv = 5, scoring = 'roc_auc_ovr', verbose = 1)
clf.fit(X, y)

L'objet clf, permet de voir tout de suite
- la meilleur valeur des parametres
print(clf.best_params_)
- le meilleur score obtenu
print(clf.best_score_)
- et le meilleur modele
print(clf.best_estimator_)
