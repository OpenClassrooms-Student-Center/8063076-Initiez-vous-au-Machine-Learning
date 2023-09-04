import pandas as pd
import numpy as np


from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=808).fit(X, y)


x = [X[8, :]]
clf.predict(x)


x = [X[13, :]]
clf.predict(x)


clf.predict_proba([X[13, :]])

clf.predict_proba([X[13, :]])


y_hat_proba = clf.predict_proba(X)

import seaborn as sns
sns.histplot(y_hat_proba[:,1])
plt.grid()
plt.title("histogramme des probabilités")

plt.savefig('histogram_bad.png')


# bad model

clf_bad = LogisticRegression(random_state=808,max_iter = 4).fit(X, y)
y_hat_proba_bad = clf_bad.predict_proba(X)[:,1]

sns.histplot(y_hat_proba_bad)
plt.grid()
plt.title("Histogramme des probabilités d'un modèle incertain")
plt.savefig('histogram_bad.png')
