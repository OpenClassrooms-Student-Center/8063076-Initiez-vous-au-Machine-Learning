# P1C3 : Découvrez les notions de modèle et d’algorithme

## Découvrez les différents types de Modèles:

    Nature des données et contexte: tabulaire, texte, sons, image, vidéo, séries temporelles: le ML s’applique à tout type de données => liste d’exemples
    Les principaux modèles  de ML: de LR (GLM) à NN/DL en passant par Tree-based;
    Pertinence en fonction du type et volume de données et aussi de leurs propriétés statistiques; dicte le choix des modèles et des outils. On distingue GLM,Tree et NN.
    Tableau comparatif.

Il existe de nombreux modeles en machine learning, à choisir en fonction du contexte.

On choisira le modele en  fonction
- de la tache a effectuer: regression, classification, clustering
- de la nature des données:chffres, categories, texte, images, vdieo, sons, serie temporelles
- du nombre d'echantillon, du volume des donnees


Le graphique suivant est extrait de la documentation scikit-learn. Bien qu'un peu daté il resume le processus de choix d'un modele.

En gros on distingue
- la regression: adapté a des jeux de donneees simples de faible taille ou les relation entre les variables sont .. lineaire. On parle aussi de generalized linear model (GLM), famille de modele qui comprend les differents types de regression lineaire.
- les modeles a base d'arbres: decision tree, randm forest, et XGBoost, adapté a des jeux de donneee plus consequent, capable de prendre en compte tout type de donnée: categoirle et robuste face au outliers et donnees manquantes. ce sont de lion les plus utilises sur des jeux de données  non enorme.
- enfin, les reseaux de neurone et le deep learning, adaptés a des jeux de donnée de gros volume de nature plus complexes comme des images ou de la video

Un aperçu de la  documentation de sklearn pour l'appremntissage  supervisé montre qu'il existe de nombreux autres modeles.
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
Certains modeles sont aussi devenu obsoletes alors que d'autres emergeaient comme plus puissant. C'est le cas par exemples des SVMs, elegants mathematiquement mais peu  competitifs  face a XGBoost.


## Sachez définir un modèle
    Encart: Fixer les idées: Modélisation ou modèle. distinguer le type  de modèle du modèle entraîné sur un jeu de donnée particulier dans un but particulier. parallèle avec la classe et l’instance en programmation.

Mais qu'est ce qu'un modele de  Machine learning? De  quoi parle-t on au final quand on parle de modele?

On va distinguer
- le type de modele: XGBoost, NN, LR: il s'agit la de la methode issue d'un raisonnement mathematique et traduite en code.
En python on importe le modele
par  exemple
```
    from sklearn.ensemble import RandomForestClassifier
```

- et son instanciation lors de l'entrainement du modele sur les données. On utilisera instance du modèle pour effectuer des predictions sur de nouveaux échantillons
En python cela consiste à utiliser la methode fit
Par  exemple
```
from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)
```
dans ce code, clf est au final le modele instancé, entraîné sur les données X et Y.


## Sachez définir un algorithme
    Définir ce qu’est un algo, illustrer avec l’algo pour calculer \sqrt(2). donner le code. Faire le lien avec le ML; mentionner le gradient stochastique

Nous avons donc un modele entrainé sur des données. Mais comment se passe cette entrainement?
Nous ne rentrerons pas ici dans le arcanes des algorithmes d'entrainement, cela serait hors sujet pour ce cours.
Mais je voudrais illustrer par un exemple, ce qu'est un algorithme. cela nous permettra de faire jaillir une notion fondamentale du machine learning: l'erreur d'estimation!

Un algorithme est une série d'instructions définies et ordonnées qui permettent d'effectuer une tâche spécifique, un calcul par exemple.

Prenons donc un exemple simple avec le calcul de la racine de 2
La méthode de Héron dite méthode babylonienne, inventé au 1er siecle apres JC par les babyloniens est une méthode efficace d'extraction de racine carrée
(je vous avais dit que le ML ne datais pas d'hier :) )


on part d'une valeur candidate (x  = 1) et on met a jour cette valeur jusqu'a ce que le resultat soit assez proche du resultat attendu.

Pour le calcul de la racine de 2: voici l'agorithme

soit une valeur d'initialisation pour x:  
x = 1
precision = 0.001
Tant que |x^2 - 2| > precision
    x = (x + 2/x)/2

le seuil permet de fixer la precision du resultat attendu.

en python cela s'ecrit

x = 1
precision = 0.001
while (abs(x**2 - 2) > precision) :
    x = (x + 2/x)/2

print(x)

Cet algorithme est extremenent efficace.
En seulement trois étapes, la précision relative sur la valeur de √2 est déjà de 10–6, ce qui est excellent, et de moins de 10–12 en quatre étapes

Remarquez maintenant la quantite |x^2 - 2|
Il s'agit de la difference en valeur absolu entre la valeur candidate au carré et 2.
c'est l'erreur d'estimation entre la  valeur candidate, notre valeur estimée et la valeur dite de vérité (groundtruth en anglais)
On peut rerecrire l'algo en mettant en avant cette erreur d'estimation

x = 1
erreur = 1
precision = 0.001

Tant que erreur > precision:
    # maj de l'estimation
    x = (x + 2/x)/2
    # maj de l'erreur avec la nouvelle valeur de x
    erreur = |x^2 - 2|

Voila, dans  ce simple algorithme qui date de 2000 ans, nous avons les composantes principales d'un algo de machine learning

- la notion d'erreur d'estimation
- la  mise a jour itérative de l'estimation

Le principal algorithme qui sous tend la plupart des modeles de ML, deep learning inclus, s'appelle  le gradient stochastique. Sans rentrer dans les  details, le GS fonctionne presque pareillement que l'algo de Héron .

Soit un jeux d'echantillons
On cherche a estimer un vecteur qui minimise un critere, une fonction de cout

On se dote d'un parametre essentiel, le learning rate \alpha qui va adapter la correction de l'estimation en fonction de l'erreur d'estimation.
Phase d'initialisation du vecteur h, par exemple que des zeros
A chaque etape
- randomizer les echantillons
- calculer l'erreur d'estimation: e
- maj de h en fonction de l'erreur: h = h + \alpha * e

le parametre \alpha sert a regler la rapidite de convergence de l'algorithme et la precision des resultats. C'est le taux d'apprentissage

- alpha petit: convergence lente mais meilleure precision
- alpha grand: convergence rapide mais resultat moins precis
- alpha trop grand: explosion et non convergence.

Illustration  

L'algo du gradient stochastique date aussi des années 50. Il existe de nombreuses variantes et il est utilisé dans la plupart des modeles de machines learning.
