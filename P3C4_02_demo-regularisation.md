

Pour montrer l'impact de la régularisation  il nous faut un dataset et un model qui overfit donc complexe et qui offre de la regularisaton de type L2. Ridge est un modele de simple regression lineaire donc par definition simple.
On peut rendre le plus complexe en l'appliquant à une régression polynomiale.

Nous l'avons brievement vu au chapitre ... la regression polynomiale consiste
à regresser la cible y non plus seulement à partir du predicteur x
y ~ x
mais à partir des puissances du predicteur
y ~ x + x^2 + ... + x^N

La complexité du modele et sa capacité à overfitter va donc croître avec la puissance du polynome en x.

Nous allons donc comparer des modeles de regression polynomiale avec differents niveaux de regularisation L2

Nous suivons les étapes:
Etape 1: mise en scène
- Créer un dataset simple à une variable predictrice avec make_regression

- Entrainer une regression lineaire simple y  ~ x avec le modele Ridge et sans regularisation
En effet, la regularisation n'est pas necessaire puisque le modele n'overfit pas. Comme on peut l'observer, il sous-performe.

Etape 2: l'overfit
Considerons ensuite la régression polynomiale de degré 12! Autrement plus complexe que la simple regression linéaire.

y = x + x^2 + ... + x^12

Nous utilisons la transformation PolynomialFeatures.

Ce modele overfit fortement le dataset.

Etape 3: la regularisation

En augmentant la quantité de régularisation ()à travers des valeurs croissantes du paramètre alpha du modele Ridge), on observe bien un effet d'attenuation de la sensibilité du modele qui ne cherche plus a passer a tout prix par tous les points du dataset. L'overfit est réduit

Même si cet exemple peu paraître artificiel, il montre bien le bénéfice de la régularisation de type L2 sur un modèle qui overfit.
