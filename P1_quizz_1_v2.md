# Notions à tester

- distinction entre modélisation stats et modélisation prédictive
- le but d'un modèle prédictif: performance et robustesse capacité d'extrapolation
- jeux de donnée: variable cible vs variable prédictives
- dataset sert à entraîner et aussi évaluer les perf
- supervisée ou non supervisée
- classification vs regression
- 3 grands classes de modèles ML classiques: régression, arbres, réseaux de neurones
- entraîner un modele: a partir de l'erreur d'estimation
- process data science: réaliser un benchmark, nécessite supervision des models en prod

# Contexte

Vous avez récemment rejoint l'office des forêts avec comme mission la détection des feux de forêts en France.
Vous décidez d'utiliser les techniques de points en machine learning.


# Q1) distinction entre modélisation stats et modélisation prédictive
Le premier jour, le ministère du tourisme vous pose une série de questions.
Lesquelles relèvent du machine learning et non de la modélisation statistique?

Est-ce que les forêts en monoculture sont plus exposées aux risques de feux de forêts que les forêts mixtes ?
Quelle est la relation entre la biodiversité d'une forêt et la quantité de lumière solaire qu'elle reçoit ?
[vrai] Peut-on établir des prévisions de la probabilité de feux de forêts, parcelle par parcelle, pour le mois prochain ?

Responses:
les questions 1 et 2 sont a propos de relations entre diverses variables et sont donc des questions de statistique
la question 3 est bien une question de prévision et donc relève du machine learning


# Q2) le but d'un modèle prédictif: performance et robustesse capacité d'extrapolation

Vous entraînez un modèle de prédiction des feux sur les données 2022 du sud de la France, en exploitant les caractéristiques des forêts méditerranéennes.
Vous appliquez ce modèle sur une région du nord du Maroc, dont les forêts sont aussi méditerranéennes. Vous observez de bonnes performances des prédictions.
A votre avis pourquoi?

- [vrai] le modèle est robuste et sait extrapoler à des données non vu auparavant a conditions que les données d'entrées soient similaires.
- Votre modèle en fait marche dans le monde entier quel que soit le climat et le type de forêt.
- si on retourne la carte de 90 degrés, l'ouest devient le sud donc le modèle est invariant par rotation

reponse:
- la question 1 est la bonne réponse. La bonne performance de votre modèle dans les 2 régions montre sa robustesse vis à vis de données nouvelles.

Le modèle peut montrer de bonnes performances sur les deux régions en raison de caractéristiques environnementales similaires, de données d'entraînement représentatives ou d'une robustesse intrinsèque du modèle face à des variations entre les régions.

- la question 2 est peu réaliste. Les forêts sont très différentes de région en région et n'auront pas les mêmes caractéristiques. Le modèle ne saura pas extrapoler à des climats et des régions très différentes des données d'entraînement. 
- la question 3 est évidemment absurde.


# Q3) jeux de donnée: variable cible vs variable prédictives
Ce jeu de données sur les feux de forêts au Portugal, a comme variable cible la surface de forêt brûlée. 
Il est disponible sur UCI a l'adresse https://archive.ics.uci.edu/dataset/162/forest+fires

Qu'est ce qui est vrai
- la variable cible indique uniquement si il y a eu un feu ou non
- [vrai] il y a 12 variables prédictives et 1 variable cible
- il y a moins d'une centaine d'échantillons

La réponse 2 est la bonne, il y a 13 variables ou attributs donc 1 variable cible et 12 variables prédictives
- la réponse 1 est fausse: la variable cible correspond à la surface brûlée, c'est une variable continue et non binaire
- la réponse 3 est fausse: il y a 517 échantillons en tout

# Q4) supervisée ou non supervisée

Soit un dataset sur les incendies de forêt avec 1247 échantillons qui comporte les colonnes suivantes
- coordonnées géographique par parcelle
- nombre d'espèces d'arbres
- altitude
- température maximum par mois sur les 12 derniers mois
- surface des feux

Quel type de modélisation pouvez-vous potentiellement réaliser à partir d'un tel dataset?

- [vrai] projet 1: réaliser une modélisation non supervisée qui regroupe des régions présentant des schémas d'incendie similaires, en fonction de la répartition des espèces d'arbres, des conditions d'altitude et de température similaires.
- [vrai] projet 2: réaliser une modélisation supervisée en créant un modèle qui prédit la surface des feux en fonction des autres variables
- projet 3: réaliser une modélisation non-supervisée qui analyse la relation de cause à effet entre la présence d'écureuil et la probabilité des feux

Reponse

Les projets 1 et 2 sont réalistes et correspondent bien respectivement à des approches supervisé et non supervisé
Le projet 3 n'est pas une modélisation non supervisée mais plutôt une étude statistique


# Q5) classification vs regression

Vous travaillez sur le même dataset qu'à la question précédente.
Vous souhaitez développer un modelée classification binaire mais il n'y a pas de variable qui soit déjà binaire.
Que pourriez-vous faire?


- prendre le log de la somme de la température sur les 12 derniers mois
- utiliser le nombre d'espèces d'arbres comme variable cible
- transformer la variable surface de feux en variable binaire en indiquant simplement si la surface brûlée est > 0 ou non

reponses:
La question 3 est vraie. considérer 2 cas exclusifs permet de construire une variable binaire
la question 1 est fausse, le log de la somme d'une variable continue reste une variable continue. de même pour tout calcul équivalent (exp, moyenne etc )
La question 2 est fausse, on pourrait considérer le nombre d'espèces d'arbres comme une catégorie (1,2,3, ...). Qui pourrait être grande. mais la ne constituerait pas une variable binaire.

# Q6) 3 grands classes de modèles ML classiques: régression, arbres, réseaux de neurones

On vous donne un dataset composé d'images satellites montrant des photos régulières des parcelles de forêts tout au long de l'année.
Le jeu de données est constitué de plus de 10.000 photos.
Vous avez annoté une centaine d'images à la main. Cela prend du temps et c'est fastidieux.
Vous voulez maintenant entraîner un modèle pour annoter automatiquement les autres
Le projet est de construire un modèle de détection de la présence ou non d'un feu sur l'image.

Qu'allez-vous utiliser comme type de modèle?
- régression linéaire: le modèle le plus simple est toujours le meilleur
- tree based: on parle d'arbres après tout ça devrait être adapté
- Les réseaux de neurones sont plus adaptés à traiter des images

réponse:
Les réseaux de neurones sont plus adaptés aux traitements d'images.
La régression linéaire est adaptée à des dataset avec peu de variables. il sera difficile d'extraire les informations des images pour les condenser dans assez peu de variables.
Les modèles à base d'arbres ne sont pas spécialement adaptés à traiter des images satellites de forêts.



# Q7) entraîner un modèle: à partir de l'erreur d'estimation
On considère le même dataset avec comme variable cible la surface des feux

Qu'est ce qui ferait une bonne erreur d'estimation pour l'entraînement d'un modèle de régression de la surface de feux

- surface prédite - surface brûlée
- [vrai] abs(surface prédite - surface brûlée )
- surface prédite / surface brûlée

La question 1 est fausse . L'erreur d'estimation doit être toujours positive de façon à pouvoir être minimisée.
en effet on ne peut trouver de minimum a une fonction qui peut toujours être encore plus négative
La question 2 est vrai
La question 3 est fausse, si il n'y a pas eu de feu, le dénominateur a zéro donnera une erreur infinie quelle que soit la prédiction.


# Q8) process data science: réaliser un benchmark, nécessite supervision des models en prod

Toujours en considérant le même jeu de donnée,
qu'est ce qui ferait un bon benchmark avant de se lancer dans une modélisation prédictive ML pour prédire les surfaces brûlées

- aucun feux donc surface prédite toujours à 0
- [vrai] prédire la moyenne des surfaces brûlées sur tout le jeu de donnée
- la surface brûlée est égale à la hauteur des arbres / nombre d'espèces d'animaux * température moyenne

Le premier benchmark bien qu'optimiste (aucun feu) est trop simple.
N'importe quel modèle aléatoire qui prédirait certaines surfaces par hasard serait meilleur que cette référence de 0.
Donc prédire 0 tout le temps n'est pas très utile

Le deuxième benchmark est aussi simple et donne au moins une information sur la quantité de feux auxquels on pourrait s'attendre. c'est la bonne réponse

Enfin le troisième est sans fondement aucun. 








