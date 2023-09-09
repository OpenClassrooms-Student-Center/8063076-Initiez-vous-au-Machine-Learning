# Nettoyez un jeu de données
    La data est souvent bruitée. Il faut la nettoyer.
    80% du temps de l'activité d'un data scientist
    -> faire référence a cours https://openclassrooms.com/fr/courses/7410486-nettoyez-et-analysez-votre-jeu-de-donnees

# Palliez les données manquantes
    Comment pallier les données manquantes : remplacer par la moyenne, les supprimer, leur donner une valeur particulière?

# Détectez les anomalies
    Détecter les outliers / anomalies et décider de la bonne approche de remédiation: suppression, cap, …

# Normalisez les valeurs numériques
    Normaliser les valeurs numériques pour éviter une trop forte disparité entre les variables.
# À vous de jouer !
    utiliser le dataset des arbres de Paris qui est très bruité et illustre presque tous les cas cités

# Améliorez un jeu de données

Nous avons jusqu'à présent travaillé sur des jeux de données simples, de faible volume et surtout très propres. Aucune anomalie ni de valeurs manquantes et les variables sont explicites.

C'est une situation idéale malheureusement peu représentative de la réalité professionnelle où l'on est confronté à des données plus chaotiques.

Je vous incite à suivre le cours https://openclassrooms.com/fr/courses/7410486-nettoyez-et-analysez-votre-jeu-de-donnees pour tout ce qui concerne l'analyse et le nettoyage des jeux de données.

Dans ce chapitre nous allons travailler sur un "vrai" jeu de données que je trouve plutôt sympathique.

Le dataset des arbres de Paris qui contient des informations sur plus de 200k arbres sur Paris intra muros et sa périphérie proche.

Les données sont mises à jour toutes les semaines, j'ai donc fait une copie pour avoir une version stable de base de travail.

Vous trouverez le dataset original sur le portail open data de Paris https://opendata.paris.fr/explore/dataset/les-arbres/information et la copie sur laquelle nous allons travailler dans le github du cours.
https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/paris-arbres-2023-09-07.csv

Ce qui est formidable avec ce dataset, hormis de pouvoir visualiser tous les arbres d'une ville comme Paris, c'est qu'il illustre les principaux problèmes qu'un data scientist peut rencontrer. A savoir:

- données manquantes
- données aberrantes ou outliers
- categories trop nombreuses ou sous-représentées
- mauvais étiquetage

Le page du dataset sur le portail opendata ne fournit pas de description des champs ni d'information sur le mode de récolte des données. Toutefois les noms des champs sont assez parlant et nous nous en contenterons.

## Données manquantes
Commençons donc par les données manquantes pour une variable donnée.

On a 3 stratégies possibles
- ignorer et supprimer tous les échantillons pour lesquels cette valeur manque. Cela vaut seulement si cela ne concerne qu'une petite partie du dataset.
- remplacer les valeurs manquantes par un valeur spécifique qui indique que la valeur n'est pas disponible. Par exemple -1 ou 0 pour des nombres ou 'none' pour une catégorie. On espère alors que le modèle saura prendre en compte l'information.
- Inférer les valeurs manquantes à partir des autres variables. par exemple prendre la moyenne,  ou la valeur médiane, etc des valeurs disponibles ou construire une régression linéaire à partir des autres variables.

Regardons ce qu'il en est sur le dataset des arbres.

Le dataset étant assez riche, on va se limiter aux Platanes. Cela nous laisse plus de 42.500 d'arbres.

df = df[df.libelle_francais == 'Platane'].copy()

la variable stade_de_developpement a 3350 valeurs manquantes (NaN)

stade_de_developpement
Adulte     	21620
Jeune (arbre)Adulte   8356
Jeune (arbre)  	5916
NaN       	3350
Mature     	3346

On suppose que la valeur "Jeune (arbre)Adulte" n'est pas un erreur de saisie mais plutôt une valeur intermédiaire entre Jeune (arbre) et Adulte. On a donc la graduation Jeune (arbre), Jeune (arbre)Adulte, Adulte puis Mature.

A ce stade on peut choisir de simplement supprimer tous les échantillons où stade_de_developpement est absent. Cela ne concerne que 7,8% des données.

On peut aussi essayer de remédier aux valeurs manquantes en observant la relation entre les mesures des arbres et leur stade de développement. Notre hypothèse est que les arbres jeunes ou Mature sont nettement plus petit que les arbres Adulte.

Le boxplot montre la répartition de la hauteur et de la circonference par stade de développement.

sns.boxplot(df = df, y="circonference_cm", x="stade_de_developpement")
sns.boxplot(df = df, y="hauteur_m", x="stade_de_developpement")

(après avoir supprimé les valeurs extrêmes avec df = df[(df.circonference_cm < 1000) & (df.hauteur_m < 100)].copy())

On observe bien une nette difference entre les arbres jeunes ou Mature par rapport aux arbres Adulte.

Donc on peut établir une règle certes arbitraire mais qui fait sens pour identifier les arbres Mature parmi les données manquantes.
Par exemple, en fixant un seuil minimum de hauteur_m > 20 et circonference_cm > 200 au vue de la figure précédente.

cond = (df.stade_de_developpement.isna()) & (df.hauteur_m > 20) & (df.circonference_cm > 200)
df[cond].shape

ce qui donne 22 arbres que l'on peut maintenant labelliser comme Mature

de même pour les jeunes arbres
cond = (df.stade_de_developpement.isna()) & (df.hauteur_m < 8) & (df.circonference_cm < 50)
df[cond].shape

ce qui donne 2903 arbres que l'on peut maintenant labelliser comme 'Jeune (arbre)'

L'étape d'après, plus complexe serait d'utiliser une régression logistique pour identifier le stade de développement à partir des variables significatives, hauteur, circonférence mais aussi par exemple la domanialité et l'espèce.

cf le notebook du chapitre.


## Les outliers

Contrairement aux données manquantes ou l'absence de valeurs sautent aux yeux, le cas des outliers est plus délicat.
Une valeur peut paraître extrême mais l'est elle vraiment ?

Une age de 200 ans est évidemment impossible (pour le moment), mais qu'en est il d'un âge de 80 ans dans un dataset de coureurs de marathon ? S'agit-il d'une erreur ou bien d'un senior en pleine forme ? Il n'y a pas de moyen d'apporter une réponse systématique à la question. Cela dépend vraiment du contexte.

Il y a 2 types d'outliers. Les données vraiment absurdes (200 ans d'âge) et les données extrêmes mais toutefois pertinentes.

Les outliers ont une influence directe sur le modèle en biaisant les statistiques de la distribution de la variable.

Prenez le cas de la moyenne de cette suite de chiffres
moyenne ([-1.2, 0.15, -0.64, 0.46, -1.26, -1.89, -0.34, 0.66, 1.48, -1.26]) =  0.17
Mais si on ajoute la valeur 10
moyenne ([-1.2, 0.15, -0.64, 0.46, -1.26, -1.89, -0.34, 0.66, 1.48, -1.26, 10]) = 1.06
Une valeur extreme peut avoir un fort impact sur la statistique et donc sur le modèle.


Visualiser la distribiution de la variable par un histogram, ou un boxplot donne déjà une bonne indication de la présence d'outliers.

Traçons par exemple, la hauteur et la circonférence des platanes. On trouve un platane de 700 m de hauteur et 2 platanes de plus de 10m de circonférence.

./figs/p3c2_01_outliers.png

On peut enlever ces échantillons sans se poser trop de questions
df = df[(df.circonference_cm < 1000) & (df.hauteur_m < 100)].copy()

Il existe aussi des techniques de calcul d'identification des outliers. Il nous appartiendra ensuite de les traiter en tant que tel ou non. Nous allons calculer un score qui traduit l'écart de la valeur par rapport à la distribution de la variable.

### Z-score
Le z-score mesure de combien d'écart-types une valeur est éloigné de la moyenne de la variable.
On considère que un zscore de 2 ou 3 correspond à un outlier.

from scipy import stats
df['z_circonference'] = stats.zscore(df.circonference_cm)
df['z_hauteur'] = stats.zscore(df.hauteur_m)

Si on plot les zscore pour la hauteur et la circonférence on observe

./figs/p3c2_02_zscore.png

de nombreux échantillons ont un zscore élevé > 2 ou > 3
Note: On observe aussi que la hauteur prend des valeurs plus discrète que continue. Cela est sûrement dû à la méthode de mesure de la hauteur des arbres. Il y a un effet d'arrondi. C'est là que l'on souhaiterait en savoir plus sur la méthode utilisée pour constituer ce dataset.

Un seuil zscore à 2 sur la hauteur (resp circonférence) détectera 793 échantillons (resp. 1429) et à 3, 33 échantillons (resp. 349)

On peut aussi regarder la méthode IQR.

L'IQR, est la différence entre le 25eme centile (Q1) et le 75eme centile (Q3) des données. Les points de données qui tombent en dessous de Q1 - 1,5 * IQR ou au-dessus de Q3 + 1,5 * IQR sont considérés comme des valeurs aberrantes.

iqr = np.quantile(df.hauteur_m, q=[0.25, 0.75])
limite_basse = iqr[0] - 1.5*(iqr[1] - iqr[0])
limite_haute = iqr[1] + 1.5*(iqr[1] - iqr[0])

Ce qui donne -5.5 pour la limite basse. pas très utile vu que les arbres n'ont pas de hauteur négative!
et 30.5 pour la limite_haute, plus intéressant. Pour la circonférence on obtient une limite haute à 305 cm.


Une limite haute pour la hauteur à 30.5 identifiera 44 arbres comme outliers. et pour la circonférence 485 arbres.

En résumé, nombre de'chantillons identifiés comme ayant une valeur aberrante
            | hauteur | circonference
zscore <  2 | 793     | 1429
zscore <  3 | 33      | 349
IQR         | 44      | 485

Donc un grand éventail de résultats selon la méthode et les seuils choisis. A ce stade le bon choix ne peut se faire qu'à partir d'une connaissance du sujet.

Note: N'étant pas un expert en arbre et foret, meme si le sujet m'interesse, j'etais bien en mal de detected des valeurs aberrantes pour la circonference des arbres. Est ce qu'un Platane eut faire 5 m de circonference? soit 1, ... me de diametre? Improbable mais on ne sait jamais. J'ai donc pris les coordonnées geographiques de arbres les plus gros (colonne geo_point_2d) que j'ai soumis dans google maps en mode street view. Et j'ai pu constater qu'il n'y avait pas de giga arbre dans ces rues.

Une fois identifiés, il peut y avoir plusieurs façons moins drastiques de traiter les outliers.


On citera:

- prendre le log de la valeur. Cela réduira fortement la dispersion de la distribution et l'influence des outliers
- Règles arbitraires: choisir de fixer une limite supérieure à la variable. Tous les arbres jeunes ont une hauteur max égale au 95e centile de la hauteur
- Discrétiser la variable par intervalle en laissant le dernier intervalle ouvert pour inclure les valeurs extrêmes.

On utilise pour cela les méthodes qcut et cut de pandas

qcut va scinder la variables en intervalles de volume sensiblement égale en fonction de leur fréquence

pd.qcut(df.hauteur_m, 3, labels=["petit", "moyen", "grand"]).value_counts()
hauteur_m
petit  17875
moyen  12568
grand  12142

et cut va scinder les données en intervalles d'ecart équivalent

pour des outliers on aura donc

pd.cut(df.hauteur_m, 3, labels=["petit", "moyen", "grand"]).value_counts()
hauteur_m
petit  40058
moyen   2524
grand	3

# Normalisez les valeurs numériques

Certains algorithmes de machine learning tels que la régression linéaire ou logistique sont sensibles à l'amplitude de valeurs des variables. Celles dont les valeurs sont plus grandes dominent le processus d'apprentissage. Les variables de valeurs plus petites sont cachées. Le modele a plus de difficultés a les exploiter et voit ses performances se degrader.

Il faut donc normaliser les variables pour qu'elles aient des amplitudes similaires.

Les 2 methodes de  normalisation les plus  courantes sont

- Min-Max Scaling
X_normalized = (X - X_min) / (X_max - X_min)
Les données sont toutes entre 0 et 1

- Z-Score Standardization (Standardization)
X_standardized = (X - X_mean) / X_stddev

Les données ont toutes une moyenne nulle et un écart type de 1

En python sur la hauteur et circonference des  arbres, cela conne:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['hauteur_standard'] = scaler.fit_transform(df.hauteur_m.values.reshape(-1, 1))
df['circonference_standard'] = scaler.fit_transform(df.circonference_cm.values.reshape(-1, 1))

fig: 

On peut aussi transformer la variable en considérant son logarithme si elle est positive. Cela va reduire fortement son amplitude sans perte d'information pour le modèle.

df['circonference_log'] = np.log(df.circonference_cm + 1)
df['hauteur_log'] = np.log(df.hauteur_m + 1)

(Note toujours ajouter +1 avant de prendre le log pour eviter les log(0))

fig: ./figs/p3c2_03_log.png

# À vous de jouer

Il se trouve que 1688 platames ont une hauteur nulle égale à 0 et 1592 ont une circonférence aussi égale à 0

En ce qui concerne la hauteur, il se peut que cela soit dû à la discrétisation observé précedemment et donc au mode de mesure utilisé.

Pour la circonférence par contre, ces valeurs nulles sont plus probablement des valeurs manquantes.

- Comment remplaceriez-vous ces valeurs nulle (=0) par des valeurs qui font sens ?

- regardez maintenant les autres principales espèces du dataset (df.libelle_francais.value counts()) comme le Marronnier ou le Tilleul. Ces espèces ont-elles des données manquantes ou des outliers? comment les traiterez vous après les avoir identifiés ?


Dans ce chapitre nous avons traités 3 des problèmes les plus fréquents que l'on peut rencontrer sur un vrai jeu de données:
- pour remédier aux valeurs manquantes nous pouvons les remplacer par des valeurs arbitraires ou en fonction des valeurs de la variable en question
- pour detecter les valeur aberrantes on utilise la methode du z-score ou de IQR
- et pour remédier aux valeurs aberrantes, nous pouvons discretiser la variable ou forcer une valeur maximum.
- enfin pour les variables d'amplitudes nous pouvons les normaliser, les standardiser ou prendre leur logarithme.
