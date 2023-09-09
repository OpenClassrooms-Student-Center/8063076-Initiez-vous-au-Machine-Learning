# P3C3 : Transformez les variables pour faciliter l'apprentissage du modèle
## Transformez des données
Transformer des données catégoriques textuelles en variables chiffrées avec le one hot encoding
## Anticipez l'impact de beaucoup de variables
curse of dimensionality: quand trop de variables noient l’information et limites du one hot encoding.
mentionner les autres types d'encoding et de la librairie dédiée à l'encoding
## Créez de nouvelles variables
le feature engineering ou l'art de créer de nouvelles variables.
## Illustration avec le dataset advertising et le carré, cube des variables.
## À vous de jouer !
Application à un cas ou une variable catégorielle avec beaucoup de valeurs avec comme conséquence directe un problème de dimension (curse of dimensionality)

---

Il ne vous aura pas échappé que les modèles n'acceptent comme données d'entrée que des chiffres. Si votre dataset d'entraînement ou vos échantillons de prédictions contiennent du texte, vous aurez une erreur python.

Et comme vous avez pu le constater dans le dataset des arbres, les données textuelles sont fréquentes.
Dans ce cours, nous n'aborderons que les categories textuelles et non pas le texte sous forme libre qui relève de techniques spécifiques appelées traitement du langage ou NLP (lien vers cours OC).

Le traitement apporté aux données catégoriques textuelles va dépendre du nombre de catégorie:

On distinguera le cas où l'on a 2 catégories, le cas binaire, et pour être très précis les cas, "un peu plus que 2" ou "carrément beaucoup".

Reloadons les donnees et comme precedemment on se limite aux platanes

filename = './../data/paris-arbres-2023-09-07.csv'
arbres = pd.read_csv(filename, sep = ';')
df = arbres[arbres.libelle_francais == 'Platane'].copy()


- Commencons par le cas binaire

La variable prend 2 valeurs distinctes et exclusives de type : oui/non, vrai/faux, mort/vivant, homme/femme, positif / negatif au test, etc

Sur le dataset des arbres la variable Remarquable est binaire.

df.remarquable.value_counts()
remarquable
NON 185232
OUI 179

Il est trivial de transformer ces valeurs en chiffres en choisissant de manière arbitraire une valeur pour chaque catégorie

df.loc[df.remarquable == NON, 'remarquable'] = 0
df.loc[df.remarquable == OUI, 'remarquable'] = 0

on obtient alors
df.remarquable.value_counts()
remarquable
NON 0
OUI 1


- cas non binaire mais peu de catégories

Les choses se compliquent un peu et l'on distingue les cas ou les catégories sont ordonnées (ordinales) ou non.

Exemple de categories ordinales:
- petit, moyen, grand
- glacé, froid, chaud, tiède, brûlant
- archi nul!, nul wesh!, super!, mega top!,

et de categories non ordonnées:
- bleu, rouge, violet, vert, gris
- épicé, salé, sucré, onctueux, amer, ...

On peut remplacer les categories ordinales par des nombres croissant sans modifier la quantitité d'information.


Supposons que le stade de développement des arbres est ordonné suivant:
Jeune (arbre) < Jeune (arbre)Adulte < Adulte < Mature

Une methode pour numériser ces catégories sera par exemple

categories = ['Adulte', 'Jeune (arbre)Adulte', 'Jeune (arbre)', 'Mature']
for n, categorie in zip(range(1,len(categories)+1), categories):
 print(categorie, n)
 data.loc[data.stade_de_developpement == categorie, 'stade_de_developpement'] = n

Et on obtiendra
stade_de_developpement
1 78454
2 38503
3 36476
4 7406
avec une  équivalence (mapping) entre les categories et les nombres
1 => Jeune (arbre)
...
4 => Mature

La libraiire category_encoders comprend un grand nombre de techniques d'encodage de categories. Cette librairie fait partie de l'ecosystème scikit-learn et respecte la meme sémantique: fit, transform etc
Vous pouvez l'installer avec

!pip install category_encoders

Pour encoder des categoriees ordinales, nous  utilisons la classe OrdinalEncoder
https://contrib.scikit-learn.org/category_encoders/ordinal.html#category_encoders.ordinal.OrdinalEncoder

OrdinalEncoder permet d'encoder des categories ordonnées ou sans ordre particulier.
Pour que l'ordre des categories soit respecté dans l'encodage, il faut donner un mapping explicite a la fonction fit comme ceci


from category_encoders.ordinal import OrdinalEncoder
mapping =[ {'col': 'stade_de_developpement',
    'mapping': {
                np.nan: 0,
                'Jeune (arbre)': 1,
                'Jeune (arbre)Adulte': 2,
                'Adulte': 3,
                'Mature': 4
                }
            } ]

encoder = OrdinalEncoder(mapping=mapping)
stade = encoder.fit_transform(df.stade_de_developpement)
stade.value_counts()
 on remplace les categories originales par leur equivalent numeriques
df['stade_de_developpement'] = stade.copy()
Et les stade de devleoppment sont maintenant numerisés
df['stade_de_developpement'].value_counts(dropna = False)
stade_de_developpement
3    21620
2     8356
1     5916
0     3350
4     3346

Notez que la forme du mappoing est un peu speciale. Elle permet d'encoder plusieurs colonnes a la fois en fournissant pour chaque colonne un dictionnaire d'equivalence.
Notez aussi que l'on peut specifier l'encodage des valeurs manquantes. ici np.nan => 0.






### cas non ordonnés

Si l'on remplace de la même façon les valeurs d'une categorie non ordonnées par une suite de nombres on introduit un ordre fictif dans les catégories.  Cette information supplementaire ordre pourrait etre utilisé par le modèle alors qu'elle ne correspond a rien de concret.
On voudrait éviter d'introduire une information arbitraire d'ordre dans les catégories.

Note: dans la pratique, remplacer les catégories par une valeur croissante ne m'a jamais vraiment posé de problème. Je n'ai jamais remarqué une dégradation importante des scores du modèle. Toutefois il y a une methode propre pour numériser des catégories non ordonnées appelé one hot encoding.

Le  principe est d'associer pour chaque valeur de la categorie une nouvelle variable binaire qui indique si la variable originale avait la valeur en questions.
Si la variable prend N valeurs distinct on intriduira donc N-1 nouvelles variables, la Nieme etant redondante.

Prenons un exemple avec nos arbres et la variable domanialité qui prends les 9 valeurs suivantes

domanialite
Alignement      35747
CIMETIERE        3121
Jardin           2038
DASCO             816
PERIPHERIQUE      603
DJS               182
DFPE               71
DAC                 7
DASES               3

Pour que l'exemple soit plus clair ne gardons que les arbres des 3 domanialités les plus fréquentes
df = df [df.domanialite.isin(['Alignement', 'Jardin', 'CIMETIERE'])].copy()
df.reset_index(drop = True, inplace = True)

On va créer 2 variables booléenne : is alignement et is_jardin telle que

is_alignement = 1 if domanialite = 'Alignement', 0 sinon
is_jardin = 1 if domanialite = 'Jardin', 0 sinon

is_alignement =  1 if (domanialite = 'Alignement') else 0
is_jardin =  1 if (domanialite = 'Jardin') else 0

Comme les catégories sont exclusives, un arbre ne peut pas être à la fois dans un alignement ou un jardin et dans un cimetière, la valeur cimetiere n'as pas besoin d'être explicité par une variable is_cimetiere. On a logiquement

domanialite = CIMETIERE si is_alignement =0 et is_jardin = 0

Tout cela est un peu lourd. Imaginez faire cela à la main pour une variable a 5, 10 ou plus de valeurs.

Cette technique qui s'appelle one_hot_encoding est implémentée dans scikit-learn dans le module preprocessing
https://scikit-learn.org/0.16/modules/classes.html#module-sklearn.preprocessing

from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
labels = enc.fit_transform(df.domanialite.values.reshape(-1, 1)).toarray()
labels.shape
(40906, 3)

donc 3 colonnes correspondant aux 3 valeurs de domanialité


Il faut maintenant intégrer 2 de ces 3 colonnes (la 3ème est redondante) et supprimer la colonne originale.
Donc un peu de manipulation de pandas.dataframe

df = pd.concat([df, pd.DataFrame(columns = ['is_alignement','is_jardin'], data = labels[:, :2])], axis = 1)



Remarquez la chose suivante:

le dataset, si l'on ne considère que les variables qui apporte vraiment de l'info a seulement 8 variables
(les variables idbase type_emplacement, ... n'ont pas de valeurs ou n'apportent aucune info pour exploitable par un modèle.)

Si l'on prend en compte toutes les valeurs de domanialité, soit 9 valeurs différentes, on ajoute au final 8 nouvelles variables . On double presque le nombre de variables.

De plus la plupart de ces nouvelles variables ont  principalement des 0
Par exmeple, la domanialité PERIPHERIQUE a seulement 5225 arbres sur un total de 207641 arbres soit 2,5% des arbres.
La colonne one hot encoded correspondant à cette valeur va être à 97.5% des zéros donc extrêmement peu informative.


De même si on considère les variables comme espèces qui a 559 valeurs differentes, la techhnique du one hot necoding ajoutera 558 nouvelles variables. la plupart truffées de 0 dans sans grande valeur.

On va donc en quelque sorte noyer les infos des 7 variables restantes dans la masse des variables dédiés à l'espèce ou aux autres variables categoriques.

Quand le nombre de variable explose, on appelle ça le curse of dimension, le piège des grandes dimensions.  Une dose finie  d'information brute est delayée sur un trop grand nombre variables chacune peu informatives. Le modele aura du mal a extraire la substantifique moelle du dataset.

En conclusion, le one hot encoding n'est réellement utilisable que lorsque le nombre de valeurs de la variable à numériser est faible par rapport aux nombre de variable utiles du dataset.



Passons maintenant aux choses serieuses!

# L'encodage binaire

Pour sortir du piège de la dimensionnalité, ma technique préférée de numérisation est l'encodage binaire disponible dans l'excellente librairie category_encoders

Prenons un exmeple de la variable domanialité  qui a 9 variables.
Le chiffre 9 en binaire s'ecrit 1001 soit 4 digits. En associant a chaque digit une variable  booleenne 0/1 on peut encoder les valeurs de cette variable a 9 valeurs sur 4 dimension au lieu des 8 du one hot encoding.
On aura par exemple
Alignement => 1 => 0000
Jardin  => 2 => 0001
CIMETIERE => 3 => 0010
etc ...
DASES  => 9 => 1001

De meme la variable especes qui a 559 valeurs distinctes pour tout le dataset (pas seulement les platanes) ne necessitera que 10 variables au lieu de 558 (2^10 = 1024).

L'avantage enorme du binary encoding sur le one hot encoding est la reduction drastique du nombre de variables necessaires pour encoder la variable categorique originale. On perds par contre la correspondance directe entre la valeur de la variable booleanne qui ne reprensent plus qu'un des digit et la categorie originale.

Avec category_encoders, lencodage binaire de la variable especes suit

from category_encoders.binary import BinaryEncoder
encoder = BinaryEncoder()
espece_bin = encoder.fit_transform(data.espece)
espece_bin.shape
(207641, 10)
espece_bin.head()
espece_0  espece_1  espece_2  espece_3  espece_4  espece_5  espece_6  espece_7  espece_8  espece_9
0         0         0         0         0         0         0         0         0         0         1
1         0         0         0         0         0         0         0         0         1         0
2         0         0         0         0         0         0         0         0         1         1
3         0         0         0         0         0         0         0         1         0         0
4         0         0         0         0         0         0         0         1         0         1


# feayture engieering

Revenons vers des choses plus artisanales que sont le feature engineering ou en français la création de nouvelles variables.

Le but du feature engineering est d'utiliser notre connaissance du domaine, du contexte, pour créer de nouvelles variables à partir des variables existantes. Cela ne suit aucune méthodologie claire et définie mais dépend de conclusions issues de l'analyse poussée des données, de la connaissance du contexte et du domaine.

On peut néanmoins mentionner comme techniques repandues:
- la mise en bucket . intervalle des données continues
- la transformation de variables existantes : puissance, inverse, racine carrée ou log, etc
- la création de variable "flag" qui indique si une autre variable est renseignée ou manquante ou tout autre caracteristique notable
- et la  compilation plusieurs variables ensemble:  par operations sur les  variables numeriques, agrégation de textes, filtres sur les images  etc

En fait l'amélioration du dataset par le feature engineering revient a faire à un véritable travail de détective à partir
- de l'analyse statistique des variables
- des connaissances du domaine
- de l'integration de dataset externes
- de l'etude precise des erreurs de prediction du modele.

Dans notre contexte urbain forestier, nous pourrions augmenter notre dataset avec des données sur l'environnement de chaque arbre  en integrant les  images de google maps obtenus à partir des coordonnées géographiques contnues dans le dataset.

Autre exemple de feature engineering. Dans le  chapitre 2 de la partie 2, nous avons construit une régression linéaire sur le dataset advertising. Pour rappel, le revenu dépend du budget de publicité dans 3 médias: tv, journaux et radio. La régression est de la forme ventes ~ tv + radio + journaux

Après analyse visuelle de la relation entre les ventes et la télé, nous avons vérifié qu'ajouter le carré  de la variable tv améliorerait significativement le score du modèle.
ventes ~ tv^2 + tv + radio + journaux


Dans le prochain chapitre nous allons revenir sur la partie modèle du binome (données - modèle) avec un concept essentiel au machine learning, l'overfit!
