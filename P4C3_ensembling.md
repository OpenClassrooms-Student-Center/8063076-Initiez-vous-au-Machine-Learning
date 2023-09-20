
Pour finir ce cours nous allons travailler sur l'apprentissage d'ensemble ou ensemble learning.

L'apprentissage d'ensemble consiste à combiner plusieurs modèles sous-optimisés pour construire un modèle global performant, stable, robuste. Ce principe est applicable a de nombreux type de modèles mais c'est surtout avec les arbres de decision qu'on va le rencontrer.

Au lieu de prendre l'avis d'un expert (rôle jusqu'ici joué par un modèle optimisé), nous nous fions au vote de nombreuses amateurs (rôle joué par les modèles peu performants). C'est du machine learning profondément démocratique! La sagesse des foules appliquée au machine learning.


Dans la pratique, l'apprentissage d'ensemble permet

- de réduire le biais - améliorer la performance du modèle
- de réduire l'overfit - assurer ses capacités d'extrapolation
- d'apporter robustesse et stabilité face aux outliers et aux bruit


- et de connaître le poids des différentes variables dans le score du modèle. Une fonction appelée feature importance qui va se révéler très utile pour sélectionner les meilleures variables de prédiction.


On considère 2 techniques d'ensemble: le bagging et le boosting.


Le bagging appliqué aux arbres de decision donne naissance au modele de foret  aleatoire; le boosting à la famille des XGBoost.
Plus délicat à optimser que les forets aléatoires, XGBoost est souvent le modele gagnant lorsque l'on travaille sur des données tabulaires.



## Principe du bagging et des foret aleatoires.


Le bagging consiste à entraîner de multiples instances d'un modèle donc les performances sont par construction faible. On appelle ce modèle de base un weak learner ou predicteur faible. Par exemple un arbre de décision fortement contraint en profondeur ou une regression linéaire.

On entraîne de nombreuses instances de ces predicteurs faibles. Chaque instance est entraînée sur une partie des données d'entraînementet sur une partie des variables predictives.

On combine ensuite les prédictions des predicteurs faibles pour former le modèle d'ensemble.
- dans le cas de la régression, on prend la moyenne des prédictions des predicteurs faibles
- dans le cas de la classification, on utilise le vote. la catégorie prédite par l'ensemble correspond à la catégorie la plus souvent prédite par les predicteurs faibles.


Appliqué aux arbres de décisions, le bagging donne le modèle Random Forest  / foret aléatoire. Ce modèle existe pour la classification RandomForestClassifier et la régression RandomForestRegressor. Le predicteur faible est ici, un arbre de décision dont la profondeur est limitée à 2 ou 3 niveaux. Nous avons vu au chapitre P4C1 que les performances de ce modèle ne sont pas très bonnes, son biais de prédiction est élevé.

On retrouve donc les paramètres d'un arbre de décision, entre autres max_depth sur lequel nous avons déjà travaillé.
A cela, se rajoutent les paramètres spécifiques à l'apprentissage d'ensemble

- n_estimators: le nombre d'arbres dans la forêt. Par défaut 100
- max_features: le nombre de variables qui sont considérées pour chaque arbre. Par défaut, la racine carrée du nombre total de variables.
- bootstrap: un booléen indiquant si chaque arbre est entraîné sur l'intégralité du jeu de données ou une partie échantillonnée. Par defaut True

Note: le bootstrap est une technique d'échantillonnage qui consiste à construire un ensemble échantillonné à partir d'un ensemble d'origine. A chaque sélection d'un échantillon l'élément est remis dans l'ensemble source. L'ensemble échantillonnée contiendra donc des doublons des échantillons de la source. Mais c'est voulu car cela offre des propriétés statistiques fortes utiles. Le bootstrap permet entre autres d'obtenir un intervalle de confiance pour chaque prédiction. (voir le notebook du chapitre)

###  le dataset
Dans ce chapitrre nous allons utilsier une dataset artificiel appelé Hastie, issue du livre Elements of Statistical Learning Ed. 2 T. Hastie, R. Tibshirani and J. Friedman. Le livre reference du machine learning. Entierement disponible en ligne.

Le dataset disponible dans scikit-learn est plus adapté pour faire des démonstrations sur les random forests et le XGBoost. que ne l'est notre dataset d'arbres.


Il consiste en 10 variables predictives, de distribution normale (Gaussienne) et d'une variable predictive binaire.

from  sklearn.datasets import make_hastie_10_2

X, y = make_hastie_10_2(n_samples=12000, random_state=808)


La variable cible est construite a partir des  predictions de la facon suivante

y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1

Si la somme des valeurs au carrée des variables predictive est superieure a un seuil alors y vaut 1, sinon y vaut 0.
Le seuil de 9.24  correspond  au median, d'une variable qui suit une distribuition appellée Chi-square. Voila pour les details. l'important est que ce dataset se prete bien aux modeles construit a partir d'arbres de decision.




## Application: démonstration de la réduction de biais


Regardons comment évolue le score de notre classifier du stade de developpement en accroissant le nombre d'arbres.

On va considérer un predicteur faible de profondeur 2 et on va faire varier le nombre d'arbres dans la foret de 1 a 120

tree_counts = [1,2,3,4,5,10,15,20,25,30,40,50, 60, 70, 80, 90, 100, 110, 120]

X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.8, random_state=8)

accuracy  = []
for n_estimator in tree_counts:
    clf = RandomForestClassifier(
        n_estimators = n_estimator,
        max_depth = 2,
        random_state = 8
        )

    clf.fit(X_train, y_train)
    accuracy.append(clf.score(X_test, y_test))

La  fonction score du RandomForestClassifier donne l'accuracy

En augmentant le nombre d'arbres, on voit que l'on atteint un plateau à partir de 50 arbres.
Après cette valeur, le score sur le test n'augmente pas.

## Démonstration de la réduction de variance

Le bagging permet non seulement de réduire le biais d'un modèle mais aussi de réduire l'overfit
Prenons comme modele de base,  un arbre dont la profondeur n'est pas contrainte et qui par conséquent overfit.
Fixons aussi le modele de base pour que toutes les variables soient prises en compte dans chaque predicteur faible.
Ce modele de base est construit pour overfitter.

Regardons l'évolution des scores sur les sous ensemble d'entraînement et de test en fonction du nombre d'arbres dans la forêt aléatoire.
A gauche l'accuracy sur les sous ensemble d'entraînement et de test et a droite la difference entre ces 2 scores
On observe que le modele a rapidement un score excellent, meme parfait sur le sous ensemble d'entraînement, mais que au fur et a mesure que l'on rajoute des arbres, le score d'extrapolation, sur les données de test continue à croitre.
Et de meme l'ecart entre les 2 scores (test et train) diminue sensiblement alors  que l'on rajoute des estimateurs dans la foret aleatoire.


Donc la random forest est un fantastique modele pour reduire l'overfit. On peut
- augmenter le nombre d'arbre
- contraindre l'arbre de decision de base en reduisant max_depth
- reduire le nombre de variable sur lequel chaque arbre  est entraîné

## Feature importance

Comme chaque arbre n'est pas  entrainné sur toute les variables, il est possible d'estimer l'importance de chaque variable

Reprenons notre dataset sur les arbres en  version numerisée

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

filename = './../data/paris-arbres-numerical-2023-09-10.csv'

data = pd.read_csv(filename)
X = data[['domanialite', 'arrondissement',
            'libelle_francais', 'genre', 'espece',
            'circonference_cm', 'hauteur_m']]
y = data.stade_de_developpement.values

X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.8, random_state=808)


et entrainons une random forest avec de  bons parametres

clf = RandomForestClassifier(
    n_estimators = 100,
    random_state = 8
    )

clf.fit(X_train, y_train)

print(clf.feature_importances_)
[0.039, 0.161 0.041 0.037 0.056 0.462 0.205]


la representation graphique de ces importance relatives montre que la variable principale est la circonference
et que l'arrondissement  est aussi important. Il serait interessant de regader de plus pres ce que cela reflete. Y a til des arrondissements moins favorables au developpement des arbres ?



---

## Le Boosting

Le randomforest est intrinsequement parallele. les arbres sont entrainés en meme temps sur des bouts du dataset.

Le boosting par contre enchaîne l'entraînement des predicteurs faibles de façon sequentielle, en se concentrant à chaque itération sur les échantillons qui ont généré le plus d'erreurs. Voilà pour l'idée générale.

Appliqué au arbres de décision, le boosting donne naissance a XGBoost qui veut dire Extreme Gradient Boosting. C'est la progéniture d'une forêt aléatoire avec un gradient stochastique.


XGBoost est arrivé avec force sur la scène Machine Learning au milieux des années 2010 et a connu un succès fulgurant en permettant aux participants de gagner de nombreuses compétitions Kaggle
https://github.com/dmlc/xgboost/tree/master/demo#machine-learning-challenge-winning-solutions

XGBoost existe en tant que librairie à part entière
https://github.com/dmlc/xgboost

et a ensuite ete integre dans scikit learn avec les modeles GradientBoostingClassifier

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

et GradientBoostingRegressor

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

. Il existe d'autre variantes de XGBoost appelées
LightGBM https://lightgbm.readthedocs.io/en/stable/

et

CatBoost https://catboost.ai/.

Ces versions se distinguent par
- leur rapidité d'entraînement
- l'optimisation de la mémoire de l'ordinateur
- la constitution des ensemble d'arbres
- le traitement des variables catégoriques
- mais surtout le nom et la signification des paramètres.
Voir à ce sujet l'excellent post https://machinelearningmastery.com/configure-gradient-boosting-algorithm/


ici nous allons utiliser  la version scikit-learn du XGBoost pour la classification :
GradientBoostingClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

Au niveau des paramètres il nous faut prendre en compte  un nouveau paramètre qui provient de la partie gradient stochastique du Gradient Boosting
- le learning rate qui dicte la quantité de correction prise en compte à chaque itération


# Comment sélectionner les paramètres d'un Gradient Boosting?

Même si l'on peut espérer d'encore meilleure performance pour le GBM que pour les Random Forests, optimiser le modèle devient plus difficile au fur et à mesure que le nombre de paramètres augmente. en effet nous avons maintenant les parametres des arbres de decision., cuex de la foret aleatioire et ce learning rate qui  fait partie des parametrs du gradient stochastique. Cela devient donc plus compliqué a regler.

Une bonne stratégie consiste à trouver d'abord la valeur optimale pour le nombre d'arbres

1. fixer,  max_depth a 3 (la valeur par défaut), et le learning rate a 0.01 (moins que la valeur par défaut)
2. optimiser le nombre d'arbres en faisant un grid search avec validation croisée sur différentes valeurs. par exemple
10, 50, 100, 200, 500

3. une fois le nombre arbres optimal trouvé, optimisez le learning rate
pour optimiser le learning rate, gardez cela en tête

- un learning rate élevé fera converger l'algorithme rapidement mais avec une erreur plus forte
- En réduisant le learning rate, vous allez ralentir la convergence de l'algorithme et en même temps faire croître l'erreur et donc croître le score, jusqu'à atteindre un plateau.

4. puis faites varier le max-depth mais à ce stade cela ne devrait pas changer grand chose.

Tout cela dépend évidemment énormément de votre contexte.

### Réduire la variance

autant  la variance n'est pas un probleme avec les foret aleatoire. rajouter des arbres ne fait pas augmenter la variance du modele. autant les boosted trees peuvent plus facilement overfitter.

Neanmoins, ces paramètres supplémentaires nous donne encore d'autres façon de réduire laoverfit
https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regularization.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regularization-py

- réduire la profondeur des arbres
- augmenter le nombre d'estimateurs faibles
- prendre un learning rate en dessous de 1.0
- réduire le max_features: nombre de features max par estimateur faible


### Demo

Voici un exemple de tuning de Gradient Boosting Classifier

Reprenons le dataset de Hastie et comparons la performance du modele en fonction des valeurs décroissante du learning rate

Cette fois ci on regarde le score sur le sous ensemble de test au fur et a mesure des iterations du modele.
On utilise une nouvelle metrique pour la classification.

Le LogLoss qui prend en compte l'ecart entre la probabilité de la prediction d'appartenir a la bonne classe et la classe reelle.
Par exemple un echantillon qui a un score de probab de 0.6 et qui  donc appartient a la categorie 1 sera penalisé par rapport a un echantillon avec un scoire de 0.9.

On peut obtenir la probabilité des predictions du modele a chaque iteration grace a la fonction staged_predict_proba
Cela donne le code suivant

X, y = datasets.make_hastie_10_2(n_samples=4000, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

Puis

learning_rates = [1, 0.6,  0.3, 0.1]
for lr in learning_rates:

    clf = ensemble.GradientBoostingClassifier(
                    n_estimators= 500,
                    max_depth= 2,
                    random_state= 8,
                    learning_rate= lr
    )
    clf.fit(X_train, y_train)

pour calculer le logloss a chaque iteration et a chaque valeur de alpha
    scores = np.zeros((clf.n_estimators,), dtype=np.float64)
    for i, y_proba in enumerate(clf.staged_predict_proba(X_test)):
        scores[i] =  log_loss(y_test, y_proba[:, 1])

Et on obtient la figure suivante


On observe que pour alpha de 1 ou 0.6, le modele converge puis diverge. son score sur le test repart a la hausse apres avoir diminué. Le modele perd sa capacité d'extrapolation.

Pour alpha = 0.1, le modele converge mais plus lentement que pour alpha = 0.3.

C'est la juste une illustration du type de comportement auquel vous pouvez vous attendre en jouant avec les learning rate du Gradient Boosting.


# Choisir le bon modèle parmi tous les modèles de scikit-learn

Pour finir, un mot sur le choix du modele. Avec scikit-learn vous avez l'embarras du choix et il peut sembler difficile de choisir le bon modèle.

Une  stratégie simple pour choisir le meilleur modèle quand  on débute  sur un nouveau dataset est de commencer par une régression linéaire pour comprendre les dynamiques entre les variables de prédiction et la variable cible. Cela vous permettra de sélectionner les meilleures variables prédictives. Et d'obtenir un benchmark des performances attendues.

### Importance du benchmark
Il est important dans tout projet de Machine learning d'avoir une idée des performances que l'on pourrait obtenir. Fixer les attentes est primordial. En effet, selon les contextes, une précision de 60% peut être un score fantastique (à peine meilleure que pile ou face) ou un score de seulement 95% pourrait entrainer des pertes lourdes pour l'entreprise (détection de fraude par exemple).

La regresison lineaire ou logistique est parfaite pour fixer ce benchmark car le modele est simple et les scores obtenues tres probablement ameliorable avec des models plus complexes / intelligents.

Une fois ce benchmark etablit, optimisez une RandomForest. Vous devriez obtenir de bien meilleurs scores qu'avec la régression. La random forest est plus simple à optimiser.

Selon les contextes, les RF sont plus efficaces que XGBoost. Cependant ce n'est pas la majorité du  temps.
Donc si les scores obtenues ne donnent pas satisfaction, il sera temps d'entraîner un XGBoost.


## En resume



## Conclusion du cours
Bravo da'voir suivi le cours jusqu'a la. J'espere qu'il vous a interessé et surotut qu'il vous a donné les elements pour continuer a decouvrir ce domaine incroyable qu'est le machine learning. Nous avons couvert beaucoup de sujets, de  dataset et de modeles mais malheurseuement j'ai du en laisser de coté. Si vous souhaitez continuer a approfondir le domaine je ne peut que vous conseiller de regarder de plus pres  le gradient stochastique. Indemodable et central a de nombruex models et technoque dont le deep learning.
Mais ussi de vous frotter au ML dans le cloud et notamment les outils d'AutoML comme vertex AI de Google qui sont d'une fficacité redoutable. Et je vous souhaites de bonnes explorations
