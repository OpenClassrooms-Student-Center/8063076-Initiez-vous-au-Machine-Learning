# !pip install palmerpenguins
# %pip install palmerpenguins

from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from palmerpenguins import load_penguins

sns.set_style('whitegrid')
penguins = load_penguins()
penguins.head()


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import FeatureUnion, make_pipeline
### To deal with missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier



k = 0
for specy in penguins.species.unique():
    penguins.loc[penguins.species == specy, 'species'] = k
    k +=1

penguins = penguins.sample(frac = 1, random_state = 808)

penguins = penguins[[ 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'species']]
penguins.dropna(inplace = True)

X = penguins[[ 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g']]
y = penguins.species

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100,
                                            random_state=0)
#
model = KMeans(n_clusters=3, init='random', max_iter=100, random_state=101, n_init = 'auto')
model.fit(X_train)
model.score
