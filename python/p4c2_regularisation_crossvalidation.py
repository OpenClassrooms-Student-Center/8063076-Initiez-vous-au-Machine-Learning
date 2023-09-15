import os
import re, json, csv
import time, datetime
from datetime import timedelta
import pandas as pd
import argparse
import glob

pd.options.display.max_columns = 100
pd.options.display.max_rows = 60
pd.options.display.max_colwidth = 100
pd.options.display.max_colwidth = None

pd.options.display.precision = 10
pd.options.display.width = 160
pd.set_option("display.float_format", "{:.2f}".format)
import numpy as np
import matplotlib.pyplot as plt
from category_encoders.ordinal import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_regression

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

if __name__ == "__main__":
    random_state = 3
    X, y = make_regression(n_samples=30, n_features=1, noise=40, random_state=random_state)
    input = np.linspace(np.min(X), np.max(X), 100)

    if True:
        # first fig

        fig = plt.figure( figsize=(9, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X[:,0], y, label=f"Données source")
        model = Ridge(alpha=0)
        model.fit(X, y)

        y_pred = model.predict(input.reshape(-1, 1))
        ax.plot(input, y_pred, label=f"Simple régression linéaire")

        ax.set_axis_off()
        ax.set_title("Simple régression linéaire")
        plt.tight_layout()
        plt.show()



        plt.savefig("./figs/p4c2_01_regularisation_ridge.png")

    if True:

        fig = plt.figure( figsize=(9, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X[:,0], y, label=f"Données source")
        model = Ridge(alpha=0)
        model.fit(X, y)

        y_pred = model.predict(input.reshape(-1, 1))
        ax.plot(input, y_pred, label=f"Simple régression linéaire")

        pol = PolynomialFeatures(12, include_bias = False)
        XX = pol.fit_transform(X)
        px = pol.transform(input.reshape(-1, 1))

        # sans regularisation
        model = Ridge(alpha=0)
        model.fit(XX, y)

        y_pred = model.predict(px)
        ax.plot(input, y_pred, label=f"Régression polynômiale degré 12, alpha = 0")

        ax.set_axis_off()
        ax.legend()
        ax.set_title("Le modèle polynomial de degré 12 overfit")
        plt.tight_layout()
        plt.show()
        plt.savefig("./figs/p4c2_02_regularisation_ridge.png")


    if True:

        fig = plt.figure( figsize=(9, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X[:,0], y, label=f"Données source")
        model = Ridge(alpha=0)
        model.fit(X, y)

        y_pred = model.predict(input.reshape(-1, 1))
        ax.plot(input, y_pred, label=f"Simple régression linéaire")

        pol = PolynomialFeatures(12, include_bias = False)
        XX = pol.fit_transform(X)
        px = pol.transform(input.reshape(-1, 1))

        # sans regularisation
        model = Ridge(alpha=0)
        model.fit(XX, y)

        y_pred = model.predict(px)
        ax.plot(input, y_pred, label=f"Régression polynômiale degré 12, alpha = 0")

        for alpha in [0.0001, 0.001, 0.01, 0.1]:

            model = Ridge(alpha=alpha)
            model.fit(XX, y)
            y_pred = model.predict(px)
            ax.plot(input, y_pred, '--' ,label=f"alpha {alpha}")

        ax.set_axis_off()
        ax.legend()
        ax.set_title("La regularisation atténue l'overfit")
        plt.tight_layout()
        plt.show()
        plt.savefig("./figs/p4c2_03_regularisation_ridge.png")
