
P1C5 : Passez d'une problématique business à la mise en production
Étape 1 : définir les spécifications à partir de la problématique business
Le processus itératif d’un projet data science, de la traduction de l’objectif business en problématique data à la mise en production dans un produit en passant par le modélisation ML.
Etape 2 : concevoir le prototype et la faisabilité du projet
On se concentre sur les étapes ML qui consistent en: collecte des données + sélection du modèle et de l’objectif + entraînement + optimisation + prédictions.
Illustration de chaque brique avec scikit-learn
Etape 3 : mettre le projet en production
MLops: Les challenges de la mise en production; drift des modèles, majs automatiques, optimisation des coûts; resilience;
quels sont les outils; le rôle du ML engineer

Du notebook au script; les plateformes de gestion des expériences, détection du drift et hosting. Le but est de montrer que le ML ne se limite pas a des notebooks python et a scikit-learn mais que derrière il y a tout un  écosystème que le  ML engineer doit connaître. sans faire peur.



Le processus itératif d'un projet de data science implique plusieurs étapes, allant de la traduction de l'objectif business en une problématique data à la mise en production d'un modèle de machine learning dans un produit. Voici les étapes clés :

    Compréhension du problème : Il s'agit de comprendre les objectifs et les besoins de l'entreprise. Le data scientist doit travailler en étroite collaboration avec les parties prenantes pour traduire ces objectifs en une problématique data spécifique.

    Collecte des données : Cette étape consiste à identifier et à collecter les données pertinentes nécessaires pour résoudre la problématique définie. Les données peuvent provenir de diverses sources, telles que des bases de données internes, des fichiers externes ou des API.

    Exploration et préparation des données : Les données collectées doivent être explorées, nettoyées et préparées pour l'analyse. Cela comprend le traitement des valeurs manquantes, l'élimination des valeurs aberrantes, la normalisation des variables et d'autres techniques de prétraitement des données.

    Modélisation : À cette étape, les techniques de machine learning sont appliquées aux données préparées. Différents modèles sont testés et évalués pour trouver celui qui répond le mieux à la problématique définie. Cela peut inclure des modèles de régression, de classification, de clustering ou d'autres approches.

    Évaluation du modèle : Une fois que le modèle est entraîné, il doit être évalué en utilisant des métriques appropriées. Cela permet de mesurer sa performance et d'identifier les éventuelles améliorations à apporter.

    Optimisation et itération : Si le modèle ne donne pas les résultats souhaités, il est nécessaire d'optimiser les paramètres, d'ajuster les hyperparamètres et de réitérer les étapes de modélisation et d'évaluation jusqu'à obtenir un modèle satisfaisant.

    Mise en production : Lorsque le modèle a été validé et optimisé, il est prêt à être déployé dans un environnement de production. Cela implique d'intégrer le modèle dans un produit ou un système existant, de mettre en place des pipelines de traitement des données et de garantir la disponibilité et la performance du modèle.

    Surveillance et maintenance : Une fois en production, le modèle doit être surveillé en continu pour détecter les éventuels problèmes ou dégradations de performance. Des mises à jour périodiques et des ajustements peuvent être nécessaires pour maintenir la qualité du modèle.

Le processus itératif de data science implique généralement des allers-retours entre ces étapes, avec une amélioration progressive du modèle en fonction des résultats obtenus et des retours d'expérience.


La mise en production d'un modèle de machine learning peut présenter plusieurs défis. Voici quelques-uns des principaux challenges :

1. Scalabilité : Assurer que le modèle puisse gérer efficacement de grandes quantités de données en temps réel. Le modèle doit être capable de fonctionner de manière performante et fiable, même avec une augmentation de la charge de travail.

2. Infrastructure et déploiement : Configurer une infrastructure adaptée pour héberger et déployer le modèle de manière efficace. Cela peut inclure des considérations telles que la gestion des ressources, le dimensionnement automatique, la résilience et l'intégration dans les systèmes existants.

3. Gestion des versions : Gérer les différentes versions du modèle pour faciliter les mises à jour et le suivi des performances au fil du temps. Il est important de pouvoir revenir à des versions précédentes en cas de besoin et de maintenir un suivi précis des modifications apportées.

4. Monitoring : Mettre en place des mécanismes de surveillance pour suivre les performances du modèle en production. Cela permet de détecter les dégradations de performance, les erreurs et les comportements inattendus, afin de prendre des mesures correctives appropriées.

5. Explicabilité : Les modèles de machine learning peuvent être considérés comme des "boîtes noires" difficiles à interpréter. Il est important de pouvoir expliquer les décisions prises par le modèle, en particulier dans des domaines sensibles où une transparence est requise (par exemple, dans les décisions médicales ou financières).

6. Sécurité et confidentialité : Garantir la sécurité des données utilisées par le modèle et protéger la confidentialité des informations sensibles. Les mesures de sécurité appropriées doivent être mises en place pour prévenir les attaques ou les fuites de données potentielles.

7. Maintenance et mise à jour : Assurer la maintenance continue du modèle en mettant à jour les dépendances, en appliquant des correctifs de sécurité et en ré-entraînant régulièrement le modèle avec de nouvelles données pour maintenir sa performance au fil du temps.

La mise en production d'un modèle de machine learning est un processus complexe qui nécessite une collaboration étroite entre les équipes de data science, de développement et d'exploitation. La compréhension de ces défis et leur gestion appropriée sont essentielles pour garantir le déploiement réussi et la performance continue du modèle en production.
