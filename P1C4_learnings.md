# P1C4 : Décryptez les différents modes d'apprentissage
## Faites la différence entre Machine Learning et Deep Learning
  Vous en avez entendu parler: Machine Learning vs Deep Learning (NN) vs Reinforcement Learning vs Online Learning etc + définitions rapides et exemples d'applications dans le monde réel;
  Dans ce cours on se concentre sur le ML supervisé pour small data (pas de NN / DL).


Récapitulons.
Les modèles servent à des tâches supervisées (classification, régression) ou non (clustering). Les 3 familles de modèles les plus usités sont: les modèles linéaires, ceux à base d'arbres et les réseaux de neurones dont le deep learning. Ces modèles sont pour la plupart entraînés grâce à des algorithmes de calcul de type gradient stochastique.

Cependant, au-delà du machine learning classique, il existe de nombreuses autres approches en machine learning. En voici un bref aperçu.

Mentionnons tout d'abord le Deep Learning, réseaux de neurones profonds, multi couches, qui sont à la source de l'envol de l'IA ces dernières années. Je vous encourage à suivre le cours d'OC sur le deep learning. En résumé, le DL traite des volumes gigantesques et des formats de données plus exotiques que des spreadsheets. Ces modèles ont des millions de paramètres et sont à la base des modèles de langues de type chatGPT que nous avons tout récemment adoptés mais aussi des avancées en traitement d'images et de vidéos.

La liste des différents types de learnings est longue et nous nous limiterons aux: reinforcement learning, online learning et semi-supervised learning.

Le **online learning** est adapté des méthodes de filtrage adaptatifs en traitement du signal. Une estimation d'un vecteur qui est mise à jour en continue au fur et à mesure de l'évolution des données. Par exemple, en domotique. l'adaptation d'un environnement (une pièce, un conduit etc ) en fonction des conditions de température ou d'humidité. Le modèle est constamment ré-entraîné. On distingue une phase de convergence, puis une phase de tracking de suivi du modèle par rapport à sa cible.

[illustration]

Dans le **reinforcement learning**, le modèle prend la place d'un agent indépendant qui interagit avec un environnement. par exemple un plateau de jeu. Au fur et à mesure de ses explorations et des actions qu'il entreprend, l'agent reçoit ou perd des points. En quelque sorte il reçoit des récompenses ou des punitions en fonction de ses actions sont but étant de maximiser les points obtenus. Il finit par "apprendre" la conduite optimale à tenir pour gagner. L'agent scinde son activité en 2 comportements: l'exploration qui lui permet d'engranger des connaissances vis à vis de l'environnement et l'exploitation pour l'accumulation de points.

[illustration]

Le semi-supervised learning
https://scikit-learn.org/stable/modules/semi_supervised.html
L'apprentissage supervisé pour la classification nécessite d'avoir un jeu de données labellisé assez important. Cela peut avoir un coût non négligeable car cette labellisation doit être réalisée la plupart du temps par un humain.
**L'apprentissage semi supervisé** permet d'entraîner le modèle sur un mix d'échantillons labellisés et non labellisés.
Une des approches, simplifiée, consiste à entraîner le modèle sur les échantillons labellisés, puis d'ajouter au dataset d'entraînement les échantillons non labellisés que le modèle classe avec certitude. Le dataset d'entraînement labellisé est ainsi complet en utilisant les prédictions de forte probabilité produite par le modèle, ceci en plusieurs étapes.


## Comprenez les évolutions récentes en Machine Learning
	et ChatGPT dans tout ça ? Le NLP, le Deep Learning et les LMs. explosion de la taille des modèles.

Récemment, un nouveau type de modèle a fait son apparition et promet de révolutionner nombre de métiers ainsi que l'enseignement.
Il s'agit de l'IA generative avec une déclinaison dans le domaine du texte avec les Large Language Models ou LMs, plus connu sous le nom de chatGPT, Bard, Claude ou LLAMA et dans le domaine de l'image avec Stable Diffusion, Mid Journey ou Dall-E.

Quand on parle LLM, on parle de milliards.
Ce sont des modèles de deep learning composés de milliards de paramètres, entraînés sur des quantités massives de données textuelles: milliards de documents et nécessitant des millions voire des milliards d'Euro en budgets d'infrastructure.

Un LLM est créé de la façon suivante:

Entraînement : Pour construire un large langage model, on commence par l'entraîner sur un énorme ensemble de données textuelles, comme des livres, des articles, des sites web, etc. L'objectif est de laisser le modèle découvrir les règles et les modèles du langage en observant des millions, voire des milliards de phrases.

Traitement de séquence : Le modèle lit les données sous forme de séquences de mots ou de caractères, en essayant de comprendre la structure et le sens du texte.

Représentation vectorielle : Le modèle transforme chaque mot ou caractère en une représentation vectorielle qui capture ses caractéristiques sémantiques et syntaxiques. On parle de embeddings.

Apprentissage : Au cours de l'entraînement, les paramètres du modèle sont ajustés pour minimiser les erreurs entre les prédictions du modèle et les données réelles.

Une fois entraîné, le modèle peut être utilisé pour générer du texte cohérent en prédisant la suite la plus probable de mots, en fonction du contexte précédent.

Les large language models sont polyvalents et peuvent être utilisés pour diverses tâches de traitement du langage naturel, telles que la traduction automatique, le résumé automatique, la génération de texte, la classification de texte, et bien d'autres.
