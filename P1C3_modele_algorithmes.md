# P1C3 : Découvrez les notions de modèle et d’algorithme

## Découvrez les différents types de Modèles:

    Nature des données et contexte: tabulaire, texte, sons, image, vidéo, séries temporelles: le ML s’applique à tout type de données => liste d’exemples
    Les principaux modèles  de ML: de LR (GLM) à NN/DL en passant par Tree-based;
    Pertinence en fonction du type et volume de données et aussi de leurs propriétés statistiques; dicte le choix des modèles et des outils. On distingue GLM,Tree et NN.
    Tableau comparatif.

Le modèle en machine learning peut prendre différentes formes, selon le type de problème à résoudre.
Il peut s'agir d'une régression linéaire, d'un arbre de décision,  d'un réseau de neurones,   ... les type de modeles  abondent

On choisira le modele en  fonction
- de la tache a effectuer: regression, classification, clustering
- de la nature des données:chffres, categories, texte, images, vdieo, sons, serie temporelles
- du nombre d'echantillon, du volume des donnees
-

Le graphique suivant est extrait de la documentation scikit-learn. Bien qu'un peu daté il resume le processus de choix d'un modele.

En gros on distingue
- la regression: adapté a des jeux de donneees simples de faible taille ou les relation entre les variables sont .. lineaire
- les modeles a base d'arbres: decision tree, randm forest, et XGBoost, adapté a des jeux de donneee plus consequent, capable de prendre en compte tout type de donnée: categoirle et robuste face au outlierss et donneees manquantes
- les reseaux de neurone, deep  learning,adapté a des jeux de donnée de gros volume de nature plkus complexes: images, video, etc ...
Il y a bien sur plein d'autres types de modeles, par exmeple les SVM  mais qui sont soit adapté a des problemqtieues et des donnees de type bien specififques ou simplement moins utilisé de nos jours etant donnée lefficacité des XGBoost ou des reseaux de neureones




## Sachez définir un modèle
    Encart: Fixer les idées: Modélisation ou modèle. distinguer le type  de modèle du modèle entraîné sur un jeu de donnée particulier dans un but particulier. parallèle avec la classe et l’instance en programmation.

Qu'est ce qu'un modele? De  quoi parle t on quand on parle de modele.
En langue courante un modele est Ce qui est donné pour servir de référence, de type (Larousse)

En ML on va distinguer
le ttype de modele: XGBoost, NN: il s'agit la de la methode mathematique / logicielle qui va effectuer la tache en consideration
et son instanciation apres entrainement. il s'agit la du modele entraine sur un jeu de donneee  que l'on va pouvoir utiliser pour faire des predictions sur de nouveaux echantillons
On peut faire le parallele avec la programmation
On considere une classe: le modele en tant que methode et son instanciation lemodele entrainé et pret a faire des predictions




## Sachez définir un algorithme
    Définir ce qu’est un algo, illustrer avec l’algo pour calculer \sqrt(2). donner le code. Faire le lien avec le ML; mentionner le gradient stochastique

Donc nous avons des modeles qui sont entrainé sur des enchantillons
Comment se passe ceette entrainement?
Nous ne rentrerins pas ici dans le arcanes des algorithmes d'entrainement, out of scope
mais je voudrais illustrer par un example simple, ce qu'est un algorithme

Un algorithmes est une série d'instructions définies et ordonnées qui permettent de résoudre d'effectuer une tâche spécifique.

prenons un exemple simple avec le calcul de la racine de 2
la méthode de Héron ou méthode babylonienne est une méthode efficace d'extraction de racine carrée
1er siecle apres JC, babylonien

on part d'une valeur candidate: x = 1
apres on repete les instructions suivantes

soit une valeur pour x
Tant  que |x^2 - 2| > precision
    x = (x + 2/x)/2

en python cela s'ecrit

x = 1
precision = 0.001
while (abs(x**2 - 2) > precision) :
    x = (x + 2/x)/2
print(x)


En seulement trois étapes, la précision relative sur la valeur de √2 est déjà de 10–6, ce qui est excellent, et de moins de 10–12 en quatre étapes
