# P2C2 : Entraînez un modèle de régression linéaire


## Découvrez le principe de la  régression linéaire
    Questions abordées par une régression: variable continue: prix, poids, etc
    Modèle qui allie robustesse, simplicité et facilité d'interprétation
    régression linéaire: la meilleure droite y = ax+b qui approxime tous les points (x,y). souvenirs du secondaire contexte multivarié avec exemple concret domaines d'applications (banque, assurance, environnement ou l'approche boîte noire n'est pas légale) (math: relation avec inversion de matrices?)
    Understand
    TALKING HEAD
    +
    STATIC

Cnsiderons un nouveau dataset, advertising

Il contient des données sur le budget alloué aux publicités télévisées, à la radio et dans les journaux, ainsi que les ventes résultantes.

Variable cible: Ventes

Predicteurs

(int) TV - Budget of advertisements in TV
(int) Radio - Budget of advertisements in radio
(int) Newspaper - Budget of advertisements in newspaper

Nous allons essayer de predire le volume de vente en fonction du budget publicitaire en TV, Radio, et Journaux

C'est un dataset classique issue du livre reference .... et pour l'actualiser nous pourrions remplacer TV, Radio et Journaux par
reseaux sociaux Facebook, TikTok et Twitter

La variable cible est continue, (oppose a categoriel) docn nous somme dans une logique de regresison
La régression linéaire est l'un des concepts fondamentaux en machine learning et peut etre le plus simple modele mais aussi extrement puissant. Sa simplicite lui confere des proprietes precieurse: interpretatbilité, facilite d'implementation, hyper rapidité, peu de memoire etc etc Ne vous laissez pas tromper par sa simplicité.

C'est une méthode puissante qui nous permet de modéliser et de prédire des valeurs continues en fonction d'autres variables.
Pensez-y comme une ligne droite qui relie les points dans un nuage de données, capturant ainsi la tendance générale des observations.

La regression lineaire est plus un outil de modelisation statistiqe
mais cela fait aussi un modele de ML assez simple pour entrainer notre premier modele.

En python :


Dans la suite nous utiliserons le RMSe comme metrique de  performance



## Comprenez les limites de la régression linéaire
Conditions d'applicabilité: décorrélation entre les variables et  relation de linéarité
encart: rappel sur la linéarité.
corrélation n’est pas causalité
Illustrer les limites de la régression linéaire sur des datasets particuliers


La regression lineaire necessite cependant de prendre quelques precautions.
La relation entre la variable coble et les predictuers est censé etre lineaire
Utiliser un LR dans des cas obviously non lineaire ne marchera pas

Une autre precaution utile est de verifier que les amplitudes des variables sont compatibles/.
Prenons le cas de la prediction du prix de vente d'une maison.
Si nous avons comme predictuers le nombre de m2 qui varie de 20 a 500 et ....

pour égaliser l'importance relative des différentes variables d'entrée et faciliter la convergence de l'algorithme d'optimisation, car cela évite que des variables avec des échelles différentes aient un impact disproportionné sur les coefficients de régression.


## Evitez de tomber dans le piège de la corrélation

Vous etes probablement familier avec le fait de ne pas confondre correlation et causalité
Ce n'est pas parce que 2 variables sont correles que l'une implique l'autre
Et ce n'est pas parce que le modele de LR converge, on trouve une droite que cela donne lieu a une interpretation

Pour montrer cela

Le quartet d'Anscombe est un ensemble de quatre jeux de données, chacun constitué de 11 points, qui partagent des statistiques descriptives presque identiques, mais qui diffèrent grandement lorsqu'ils sont représentés graphiquement. Ces jeux de données ont été créés par le statisticien Francis Anscombe en 1973 pour illustrer l'importance de la visualisation dans l'analyse de données et pour mettre en évidence le fait qu'une analyse basée uniquement sur des statistiques résumées peut conduire à des conclusions erronées.

Les quatre jeux de données du quartet d'Anscombe sont les suivants :

    Dataset I : Une relation linéaire parfaite.
    Dataset II : Une relation également linéaire, mais avec une valeur aberrante.
    Dataset III : Une relation non linéaire.
    Dataset IV : Une relation linéaire avec une forte influence d'un point extrême.


Évaluez les performance d'un modèle de régression
Quelles sont les métriques d'évaluation d'une régression linéaire: RMSE MAE définition, RMSE relative souvent plus parlante
Evaluate

## À vous de jouer !
Exercice de régression sur le dataset advertising. Ajouter les puissances des variables (x^2, etc ) pour introduire la régression polynomiale. teasing sur apprentissage
Apply
A VOUS DE JOUER
