Pour finir ce cours nous allons voir une technique magique appelé appretissage d'ensemble ou  ensembling en Anglais.

Cette technique fonctionne avec la plupart des modeles mais est particulierement efficace avec les arbres de decision.
Ele donne naissance a un nouveau type de modele, les forets aléatoires.
On se croirait dans de la science fiction des annees 70'!


L'idee de l'ensemblage / bling est simple

Combiner les  predictions de plusieurs modeles de predictions sous optimisés pour construire un modele tres performant.

L'idé est profondément démocratique

Au lieu de prendre l'avis d'un expert (rôle joué par un modèle hyper optimisé), vous choisissez de vous fier au résultat du vote de nombreuses personnes qui sont loin d'être des expertes dans le sujet (rôle joué par les modèles peu performants).

Dans la pratique du machine learning, cette apprentissage d'ensemble ou ensemble learning, est très efficiace.

Au dela de l'optimisation des scores de predictions, la technique d'ensemble permet aussi

- de réduire l'overfit
- de réduire le biais
- d'etre robustes face aux outliers et aux bruit
- de connaitre le poids des differentes variables dans le score du model. Une ... appelé feature importance qui est très utile.

on va donc

- choisir un modele faible, appelé weak learner
- selectionner de facon aleatoire un sous ensemble des données d'entrainement
- selectionner une partie des variables d'entree
- entrainer ce modele sur ce sous ensemble de données et de variables

On obtient donc N modeles qui on vu chacun une partie des données d'entraînement et qui sont capables de faire desprediction a partir d'une partie des variables predictives.

A ce stade, on va combiner les predictions de ces modeles
- soit en prenant la moyenne des predictions dans le cas de la regression
- soit par un systeme de vote dans le cas de la classification. La categorie étant  la plus prédite par les weak learners est la catégorie prédite.

On appelle cela la technique du bagging. mise dans un sac.

Appliqué aux arbres de décisions, le bagging donne le modèle Random Forest qui existe en version classification RandomForestClassifier et régression RandomForestRegressor
Ici le weak learner sera un arbre de decision fortement contraint. Par exemple en limitant sa profondeur a 2 ou 3 niveau.

On retrouve les parametres d'un arbre de decision,  entre autre max_depth sur lequel nous avons deja travaillé.

Mais l'on pourra aussi regler entre autre
- n_estimators: le nombre d'arbre dans la foret. Par defaut 100
- max_features: le nombre de variables qui sont considérées pour chaque arbre. Par defaut la racine carrée du nombre total de variables
- bootstrap: un boolean indiquant si chaque arbre est entrainé sur l'integralité du jeu de données ou une partie échantillonnée


Note: le bootstrap  est un technique d'ehcantillonge qui consiste a construire un ensemble echantilloneé a partir d'un ensemble d'origine. A chaque selection d'un echantillon l'element est remis dans l'ensemble source. L'ensemble ehcnatillonée contiendra donc des doublons des ehcantillons de la source. Mais c'est voulu et offre certaines proprietes satistique fortes utiles.

En répetant une experience sur un bootstrap du jeu de données initial, on obtient un interval de confiance pour chqaue prediction.


Regardons comment evolue le score de notre classfiier en accroissant le nombre d'arbres.

On va considerer un weak learner de profondeur 3. nous avons vu au chapitre P4C1  que les perforamnce  de ce modele ne sont pas tres bonnes.

code + fig

En augmentant le nombre d'arbres on voit que l'on atteint un plateau a partir de N = ... .
Apres cette valeur le score sur le test n'augmente pas

---

Cela nous amene maintenant a aborder la championne de tous les modeles
la lionne face a la gazelle (regresion lneaire),
le modele qui gagne  toutes les competitions Kaggle depuis 2018,
la progeniture d'une foret aleatoire avec un gradient stochastique
j'ai nommé le XGBoost!

Le XGBoost implemente une autre forme d'appretissage ensembliste  qu'est le boosting.
Le boosting lit: to boost: soulever ou élever en poussant par l'arrière ou par le bas.
consiste a enchainer les etapes d'entrainement des weak learners aggrégés en random forests
en se concentrant a  chaque iteration sur les echantillons qui  ont genere le plus d'erreurs

Le XGBoost existe  en pluseirs versions (LightGBM, CatBoost, scikit-learn's GBM ) toutes aussi efficiaces mais qui vont se distinguer par
- leur rapidité d'entrainement
- l'utilsiationd e la memoire
- la facon de constituer les ensemble d'arbres
- la facon de traiter les variables categoriques
- la denomination et  signification des  parametres. Voir a ce sujet l'excellent post https://machinelearningmastery.com/configure-gradient-boosting-algorithm/

ici nous allons utiliser  la version scikit-learn du XGBoost : GBM
GradientBoostingClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

Au niveau des parametres nous allons considere donc un nouveau paramatre
- le learning rate qui dicte la quantité de correction prise en conpte a chaque  iteratioin

Optimiser le modele de=vient  plus difficile au fur et a mesure que le nombre de parametre augmente.

Une bonne stratégie consiste a trouver d'abord la valeur optimale pour le nombre d'arbres

1. fixer,  max_depth a 3 (la valeur par defaut), et le learning rate a 0.01 (moins que la valeur par defaut)
2. optimiser le nombre d'arbres en faisant un grid search avec validation croisee  sur differentes valeurs. par exemple
10, 50, 100, 200, 500

une fois le nombre arbres optimal trouvé, vous vous attacherai a optimiser le learning rate

puis a faire varier le max-depth mais a ce stade ca ne devrait pas changer grand chose.

tout cela depend enormement de votre contexte.

pour optimiser le learning rate, gardez cela en tete

un learning rate elevé fera converger l'algorityhme rapidement mais avec une erreur plus forte
en reduisant le learning, vous allez ralentir la convergence de l'algorithme et en meme temps faire decroitre l'erreur et donc croitre le score, jsuqu'a atteindre  un plateau.  
