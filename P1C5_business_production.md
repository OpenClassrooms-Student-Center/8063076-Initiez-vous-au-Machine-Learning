# P1C5 : Passez d'une problématique business à la mise en production
## Étape 1 : définir les spécifications à partir de la problématique business
    Le processus itératif d’un projet data science, de la traduction de l’objectif business en problématique data à la mise en production dans un produit en passant par le modélisation ML.
## Etape 2 : concevoir le prototype et la faisabilité du projet
    On se concentre sur les étapes ML qui consistent en: collecte des données + sélection du modèle et de l’objectif + entraînement + optimisation + prédictions.
    Illustration de chaque brique avec scikit-learn
##Etape 3 : mettre le projet en production
    MLops: Les challenges de la mise en production; drift des modèles, majs automatiques, optimisation des coûts; resilience;
    quels sont les outils; le rôle du ML engineer
    Du notebook au script; les plateformes de gestion des expériences, détection du drift et hosting. Le but est de montrer que le ML ne se limite pas a des notebooks python et a scikit-learn mais que derrière il y a tout un  écosystème que le  ML engineer doit connaître. sans faire peur.



## Étape 1 : définir les spécifications à partir de la problématique business

Prenons un peu de recul avant de clore cette partie plutôt théorique.

Un projet de data science à 3 grandes phases: conception, modélisation et production.

La phase de conception a pour but de traduire une problématique business, un besoin ou un produit en projet ML.

De façon très générique, il faut au minimum:

- des données, qui soient pertinentes,
- un sujet ou un produit qui soit proprement défini,
- montrer qu'il y a un net avantage à exploiter la modélisation prédictive au lieu d'une solution plus simple.

Une simple série de règles apporte souvent une solution plus simple et parfois d'efficacité comparable. En anglais on parle de rule based solutions.

Un projet de ML est complexe. Avant de se lancer, il faut pouvoir calculer le gain réellement apporté par une démarche ML.
Cela implique de réaliser une étude de benchmark au préalable. Cette étude préliminaire permet aussi de définir ce qui constitue le succès du projet. Comment mesurer la performance du système, la métrique, et quel score est nécessaire d'obtenir pour réaliser les objectifs du projet.

Voici quelques exemple de benchmark:

- Prédiction météo: on prédit que le temps du lendemain sera le même que la veille. C'est simple et cela marche souvent bien dans certaines latitudes mais n'a évidemment que peu de dimension vraiment prédictive.
- prédiction du prix d'une course de taxi: la distance * prix/km donne une bonne approximation du prix final, mais ne prend pas en  compte la dimension temps, les aléas du parcours etc ...
- une estimation des ventes d'un produit basée sur la moyenne des 3 derniers mois devrait donne une estimation raisonnable des ventes futures.

- dans le cals d'une classification binaire

Dans tous ces cas, on pourra choisir comme métrique de scoring une mesure de la différence entre la valeur estimée et la valeur observée.

Par contre, les projets suivant auront du mal à voir le jour sans une bonne dose de ML

- prédire la défaillance d'une pièce ou d'un serveur, ou le risque de défaut d'un crédit
- classer automatiquement des sons ou des images
- détecter des contenus agressifs ou des fakes news sur les réseaux sociaux

La collecte préalable des données va permettre de s'assurer de leur disponibilité.
Au-delà des contraintes apportées par les règlements européens (RGPD) et français (CNIL), les données bancaires, de santé ou relatives aux savoir-faire industriels seront plus difficiles d'accès que d'autres.
Enfin, la propriété des données doit aussi être prise en compte. Des données scrapées d'un site ne pourront constituer la base d'une offre commerciale. Donc de nombreux obstacles avant de pouvoir entrainer le moindre modèle.

Enfin, une fois ces données collectées, il faut s'assurer qu'elles soient bien exploitables.
- Y a-t-il du signal dans les échantillons?
- Sait-on ce que représente réellement les variables
- Quelles est la couleur du cheval blanc d'Henri 4?

Donc tout un travail en amont pour pouvoir aboutir enfin à l'étape de modélisation et de machine learning.

## Etape 2 : concevoir le prototype et la faisabilité du projet

Une fois fixée une version du jeu de données, et une indication de benchmark de performance à dépasser, le but de l'étape de machine learning est d'obtenir un modèle qui soit

- performant: bon score vis à vis de la métrique choisie
- et robuste: stable face à des nouvelles données. En fonction du contexte, on pourra privilégier un modèle moins performant mais plus résilient face aux variabilités des données qu'un modèle plus performant mais plus sensible aux variations.

Les étapes de machine learning vont constituer en une série d'itérations des étapes suivantes:

- travail de mise en forme de la data
	- data cleaning: résoudre les outliers et les données manquantes
	- feature engineering: créer de nouvelles variables à partir de variables existante (prendre le carré ou le long d'un prix)
	- numérisation des données catégoriques, textuelles ou images pour être ingérable par un modèle
- choix du type de modele: GLM, Tree, NN ou autre
- stratégie de repartition des données avec une partie réservée pour l'entrainement et l'autre pour la validation
- optimisation des parametres du modele

Nous reviendrons en détail sur ces différentes  étapes dans les prochains chapitres.
Il s'agit d'un processus itératif. avec des va et viens entre la colecte des données, la reprise des objectifs et leur quantification,
la mise en forme des données, le choix des algo etc



##Etape 3 : mettre le projet en production

Le modèle une fois optimisé a vocation à être mis en production, cad intégré dans le produit / le service.
C'est alors le travail des MLOps et des DEVOps qui vont prendre en charge la mise en production dans le cloud, la maintenance informatique et la surveillance des modèles.

Pour bien comprendre l'importance de cette étape, pensez à la mise en production de centaines voir de milliers de modèles, qui doivent être automatiquement et en parallèl:

- mis à niveau
- re-entraînés
- déployés
- surveillés

Le MLOps est la contraction de DevOps (développement et opérations) et de ML. Le rôle du MLOps consiste à opérationnaliser les modèles de ML en production.

Au-delà des tâches d'intégration et de systématisation, le MLOPs va devoir surveiller / monitorer les modèles en production.
« Rien n'est permanent, sauf le changement », il est donc probable que les conditions sous-jacentes aux données d'entraînement du modèle ne seront que transitoires. Les modèles risquent à un moment de perdre de leur efficacité et il faudra les ré-entraîner, les modifier (nouvelles variables) et les redéployer.

On parle alors de drift du modèle.

Pensez aux crises mondiales récentes, guerres, pandémies, bouleversement climatique, qui ont et continuent de chambouler de nombreux modèles de prédiction.
