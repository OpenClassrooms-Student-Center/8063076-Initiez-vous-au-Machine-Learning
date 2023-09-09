# Contraignez votre modèle
Après une première passe sur les données,  on essaie d'améliorer le modèle. modifier les paramètres du modèle de régression linéaire / GLM en jouant sur la régularisation L2. introduction de la régularisation comme réglage de la sensibilité / robustesse du modèle.
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
choix  des paramètres

----

On a beaucoup parlé jusqu'à maintenant d'optimisation des modèles sans montrer en quoi cela consiste exactement.
En effet, les modèles de régression linéaire et logistique n'ont pas de paramètres.
Dans une régression brute tout dépend des prédicteurs utilisés.
Il est donc impossible de jouer sur leurs paramètres pour améliorer leur performance.

Le modèle K-NN a un paramètre principal, le nombre de clusters, et nous avons montré que ...

Dans ce chapitre nous allons plonger dans la recherche des paramètres optimaux d'un modèle dans le but d'accroître ses performances.

Nous avons vu tout au début de ce cours, au chapitre 2, partie 1, que pour évaluer les perfs d'un modèle, on scinde le dataset en 2 sous ensembles: train et test et que l'on évalue la performance d'un modèle par son score sur le sous ensemble de test.


On peut bien entendu calculer le score du modèle sur le sous ensemble du train. Ce score (train) aura d'ailleurs tendance à être meilleur que les scores(test). En effet, entraîner le modèle consiste à lui faire apprendre la dynamique (patterns) dans les échantillons d'entraînement. On suppose que ces dynamiques internes aux données seront similaires sur le sous-ensemble de test ou une fois en production sur les données réelles qu'il devra prédire. C'est l'hypothèse de base de toute le machine learning.
La similarité des distributions au sens statistique des sous-ensemble d'entraînement de test et de production.

Ces 2 scores sur les données d'entraînement ou de test vont nous permettre d'identifier les 2 cas où le modèle n'est pas satisfaisant et celui où le modèle est bon.

Dans la suite on note score(train) et score(test) les scores du modèle sur les données d'entraînement et de test.

1. Le modèle est simplement mauvais

Il n'arrive pas  a comprendre les données d'entraînement
Cela se traduira directement par un mauvais  score(train).
Si les données d'entraînement sont incomprises alors par conséquence directe, les données de test seront aussi incomprise
donc le score(test) sera faible
On parle dans ce cas de biais du modèle.

2. Le modèle est bon et sait généraliser

Dans ce cas, les 2 scores score(train) et score(test) seront bons. Your job is done! Prochaine étape production!

3. Le modèle apprend mais ne peut pas généraliser

Ce qui se traduit par un bon score(train) mais un mauvais score(test)

On appelle cela l'overfit, le sur-apprentissage. le modèle colle trop aux données d'apprentissage et ne sait pas extrapoler aux données non déjà rencontré.
Nous allons voir comment détecter l'overfit et le corriger.

Le quatrième cas, bon score(test) mais mauvais score(train) reflète surtout une anomalie statistique et n'arrive normalement pas ou très peu.


L'optimisation du modèle par le jeu paramètres aura pour unique but de trouver le juste milieu entre le biais et l'over fit.

Nous allons illustrer tout cela ainsi que les remèdes potentiels pour pallier les faiblesses du modèle en biais ou overfit.

Mais avant, fourbissons nos armes. Il nous faut un dataset et un modèle qui dépendent de paramètres.

Pour le dataset, nous reprendrons les arbres de Paris mais dans une version déjà numérisée et nettoyée.
Le code prend trop de place ici, il est disponible dans le notebook du chapitre.

Nous allons essayer d'entraîner un modèle qui prédit le stade de développement de l'arbre. n
Un cas de classification a 4 catégories que nous avons déjà vu: ['Jeune (arbre)', 'Jeune (arbre)Adulte', 'Adulte', 'Mature']

En ce qui concerne le modèle, nous allons anticiper sur la partie 4 du cours et utiliser un arbre de décision comme classificateur.
Le modele DecisionTreeClassifier de scikit-learn
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

Pourquoi un arbre de décision ? simplement parce que ce type de modèle se  prête parfaitement à la démonstration d'overfitting et de biais. Les modele lineaire de  regression sont incapable d'overfitter autrement que sur des jeu de  donneees jouets, et le KNN serait peu réaliste dans notre contexte.

Un arbre de décision est simplement un enchaînement de règles de classification qui sont établies automatiquement à partir des variables prédictrices.
On peut représenter un arbre graphiquement comme un arbre la tête en bas. Pour l'instant nous nous contenterons de cette introduction simple.

Voici un exemple simple

./img/decision_tree_example.png

Dans notre cas, les nœuds de décision correspondant aux règles de décisions se feront sur les statistiques et les mesures des variables. Par exemple
si hauteur > 10  & libelle = platane => stade de développement = Mature
L'algorithme d'entraînement détermine  les différentes règles (variable, seuils, valeurs catégoriques) les plus efficaces par rapport  à des critères prédéfinis (le paramètre criterion: The function to measure the quality of a split) .

Une caractéristique saute au yeux, le nombre de règles de l'arbre autrement dit la hauteur de l'arbre (on parle plutôt de profondeur vu que l'arbre est a l'envers).

Un arbre peu profond (pruning en anglais ou élagage) a peu de règles tandis qu'un arbre profond aura beaucoup de règles.

img/decision_tree_pruned.png
img/decision_tree-deep.png

Dans scikit-learn le paramètre s'appelle max depth.
Et c'est avec ce paramètre que nous allons jouer pour montrer le biais et l'over fit.

# Tout d'abord le biais.

Loading les données et importons le classificateur

df = pd.read_csv ....

Nous utilisons le DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
Limitons la profondeur de l'arbre a 3.

clf = DecisionTreeClassifier(
	max_depth = 3,
	random_state = 808
)


splittons les donnees en train et test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
	X, y, train_size=0.8, random_state=0
)

fittons le modele
clf.fit(X_train, y_train)
et regardons le score. on utilise l'AUC

train_auc = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
test_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
print("train",train_auc)
print("test", test_auc)

et la matrice de  confusion

y_train_hat = clf.predict(X_train)
y_test_hat = clf.predict(X_test)
print(confusion_matrix(y_test, y_test_hat))

et le rapport de classification qui donne plusieurs score
print(classification_report(y_test, y_test_hat))

precision	recall  f1-score   support

1   	0.82  	0.70  	0.76  	7272
2   	0.54  	0.50  	0.52  	7654
3   	0.79  	0.85  	0.82 	15490
4   	0.62  	0.75  	0.68  	1462

accuracy                       	0.73 	31878
macro avg   	0.69  	0.70  	0.69 	31878
weighted avg   	0.73  	0.73  	0.73 	31878

On voit que les précision (le ratio de bonne pioche parmi tous les positifs) pour les différentes classes  ne sont pas très bonnes surtout pour les catégories
'Jeune (arbre)Adulte', Mature' (resp. 0.54 et 0.62, à peine meilleur qu'un pile ou face)
et un recall de 0.5 pour la catégorie  Jeune (arbre) (le modèle n'identifie que la moitié des arbres)

ainsi qu'une accuracy de
clf.score(X_test, y_test)
0.729

Donc un modele mweh

C'est normal c'est ce que nous voulions.
Notez cependant que le score sur la partie test est similaire au score sur la partie train
clf.score(X_train, y_train)
0.731


Quels remèdes pour minimiser le biais du modèle?

2 voies sont à explorer

- modifier les paramètres du modèle pour améliorer sa performance.
- ajouter des données. Sur un dataset trop petit, le modèle n'aura pas assez d'exemples pour assimiler  les dynamiques internes. Ajouter des données pourra l'aider.
On peut soit collecté plus de données et les ajouter au dataset
soit utiliser des techniques d'augmentation de données qui  créent des échantillons artificiels et gonflent donc artificiellement le dataset d'entraînement. Voir a ce titre ...

- le fameux feature engineering ou l';on va s'efforcer de transformer ou ajouter des variables au dataset pour coder plus d'information exploitable par le modèle.

Tout cela dépend évidemment du contexte dans lequel vous travaillez.

# L'overfit

reprenons maintenant notre arbre de décision sans limiter sa profondeur. \
Pour cela on set max_depth = None


clf = DecisionTreeClassifier(
	max_depth = None,
	random_state = 808
)
le modèle est en effet meilleur on passe de 0.73 à 0.81

clf.score(X_test, y_test)
0.81

mais on  remarque  que sur le sous ensemble d'entraînement on a carrément 0.94!
clf.score(X_train, y_train)
0.935048231511254

Nous sommes bien dans un cas d'overfitting ou le modèle colle aux données d'entraînement. et ne sais pas reproduire la même perf sur les données de test.

Donc à un moment, entre maxdepth = 3 et max+depth = infini, les scores(train) et score(test) ont divergé.

Essayons de trouver a quel moment cela est arrivé en faisant croître maxdepth et en enregistrant les 2 score pour chaque valeur,
l'auc est plus parlant pour cette démo

scores = []
for depth in np.arange(2,  30, 2):
	clf = DecisionTreeClassifier(
    	max_depth = depth,
    	random_state = 808
	)
	train_auc = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
	test_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
	scores.append({
    	'max_depth': depth,
    	'train': clf.score(X_train, y_train),
    	'test': clf.score(X_test, y_test),
	})

scores = pd.Dataframe(scores)

on obtient la figure suivante

figs/p3c3_02_overfit.png

Que voit on ?

en abscisse max_depth, en ordonné, AUC

Quand on augment max_depth, l'AUC(train) croit jusqu'à presque atteindre un score parfait de 1
ce score sur train atteint un plateau indiquent qu'augmenter la profondeur / la complexité de l'arbre ne sert plus a rien a partir de max_depth = 15, 16

l'AUC sur  test par contre croît jusqu'à atteindre un maximum autour de 0.94 pour max_depth = 10
Elle décroît  ensuite jusqu'à 0.87 pour rejoindre elle aussi un plateau vers max_depth = 20

On voit bien les  3 cas de comportement du modèle
A gauche, le modèle sous performe
A droite il overfit
Au milieu on obtient la meilleure performance sur le test set.
On a trouvé la valeur optimale du paramètre max_depth pour ce model et ce dataset donné

Comment remedier a l'overfit

Un modèle qui overfit est un modèle trop complexe.
Dans notre contexte complexité = profondeur de l'arbre.
dans le cas d'une régression, complexité = trop de variable prédictives

La première stratégie de remédiation sera d'augmenter la taille du dataset d'entraînement. En effet plus il y aura d'échantillon plus la complexité du modèle sera utilisé pour comprendre la masse d'information

Mais cela n'est pas toujours possible.

# regularization d'un modele

La régularisation est une technique qui permet de tempérer les ardeurs d'un modèle.
Elles se présentent sous la forme d'un paramètre que l'on peut régler à souhait.

Pour mieux comprendre revenons a la regression lineaire mais cette fois avec ce composant de régularisation.
Dans scikit learn nous avons le modèle Ridge qui est simplement une régression linéaire avec de la régularisation
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge

La documentation de scikit-learn pour le modèle Ridge indique

Minimizes the objective function: ||y - Xw||^2_2 + alpha * ||w||^2_2

Nous avons vu que définir la fonction de coût permet  de fixer l'objectif d'apprentissage au modèle.
Dans la régression linéaire simple la fonction de coût est
||y - Xw||^2_2,  ou y est la variable cible, X la matrice des predicteurs et w représente le vecteur de coefficient de la régression linéaire. la fct de coût est donc la norme quadratique de l'erreur  d'estimation

Pour RIdge on ajoute un terme alpha * ||w||^2_2 ou alpha est le paramètre du modèle que nous pouvons regler et ||w||^2_2 jour le rôle d'une contrainte sur les coefficients du vecteur de coefficients de la régression.
Ce terme quand alpha > 0 empêchait les coefficients d'être trop disparate, de diverger. Il lie les coefficients entre eux.
Le modèle ne va donc pas pouvoir accommoder les variations les plus subtiles du dataset d'entraînement et sera forcé de trouver un juste milieu.
C'est cette quête du juste milieu qui compense l'overfitting.

Ce terme de régularisation apparaît soit avec la norme quadratique L2 soit la norme de premier degré L1
le terme de régularisation de la fonction de coût sera alors  de la forme

alpha * |w| ou |w| est la somme de la valeur absolue des coefficients du vecteur w.

Voilà pour la théorie de la régularisation.
Ce qu'il faut en retenir c'est que pour les modèles qui offrent ce mode de contrainte sur l'apprentissage,faire  en sorte d'avoir une régularisation (alpha > 0) permet dans la plupart des cas de remédier à la overfit.

Note: dans certains modèles, comme les modeles a base d'arbres ou les réseaux de neurones la régularisation prendre une autre forme que ce terme alpha*||w||^2_2. mais le principe sera le même.

Pour les arbres de décision, régulariser le modèle consiste à limiter la profondeur de l'arbre ou à jouer sur d'autres paramètres que  nous verrons dans le prochain chapitre.

Enfin et pour clore ce chapitre déjà bien long, il nous faut parler de la validation croisée, mentionnée au chapitre ... partie 1.


# validation croisee

Quand on split le dataset en test et train pour évaluer les perf du modèle, et sélectionner les meilleurs paramètres, nous sommes dépendant de la répartition des échantillons entre les datasets de test et de train.

Il se pourrait par un hasard fortuit que certains échantillons pathologiques se retrouvent dans le sous-ensemble de tests. Imaginez par exemple que dans notre Datassette des arbres de Paris, la plupart des arbres au stade matures se retrouvent dans la partie test. Le modèle ne verrai que très peu de ses échantillons de cette catégorie et donc serai bien incapable de prédire des arbres Mature.

Il faut donc trouver un moyen de s'affranchir de ces anomalies potentielles de répartitions des échantillons entre test  et train set.
Pour cela on va implémenter la validation croisée.
- nous allons diviser le dataset  en K sous-ensembles de taille égales
- et à tour de rôle chaque sous ensemble jouera le rôle de sous ensemble de test, les K-1 autres sous ensembles serviront à entraîner le modèle.
- pour chaque configuration (entraînements , test) on va calculer le score de performance en fonction des paramètres que nous souhaitons sélectionner
- et à la fin on choisira le paramètre qui offre la meilleure moyenne des scores

img/cross-validation-k-fold.png

La validation croisée ne sert pas qu'à sélectionner les meilleurs paramètres pour une dataset et un modèle donné.
Elle est utile lors de toute expérience (modification des variables, choix de modèle, ...) qui donne lieu à une décision / une sélection.

scikit-learn offre la methode  KFold

from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
for train, test in kf.split(X):
	print(f" {train.shape} {test.shape} {train[:3]} -> {train[-3:]}  {test[:3]} -> {test[-3:]}")

Il y a de multiples façons de faire  de  la validation croisée dans scikit learn. C'est même un peu difficile de s'y retrouver.
Ma méthode préférée, un juste milieu entre automatisation total et implémentation manuelle complète est GridSearchCV
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':np.arange(2,  30, 2)}
model = DecisionTreeClassifier(
	random_state = 808
)

clf = GridSearchCV(model, parameters, cv = 5, scoring = 'roc_auc_ovr', verbose = 1)
clf.fit(X, y)

print(clf.best_params_)
print(clf.best_score_)
print(clf.best_estimator_)
