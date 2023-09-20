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
import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    from  sklearn.datasets import make_hastie_10_2

    X, y = make_hastie_10_2(n_samples=12000, random_state=808)

    X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.8, random_state=8)

    if False:

        tree_counts = [1,2,3,4,5,10,15,20,25,30,40,50, 60, 70, 80, 90, 100, 110, 120, 150, 200]

        accuracy  = []
        for n_estimator in tree_counts:
            clf = RandomForestClassifier(
                n_estimators = n_estimator,
                max_depth = 2,
                max_features = 3,
                random_state = 8
                )

            clf.fit(X_train, y_train)
            accuracy.append(clf.score(X_test, y_test))

            print(n_estimator, clf.score(X_test, y_test), clf.score(X_train, y_train))


        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(tree_counts, accuracy)
        plt.plot(tree_counts, accuracy,'*')
        ax.grid(True, which = 'both')
        ax.set_title('Accuracy on test vs n_estimators')
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.9 * np.min(accuracy), 1.1 * np.max(accuracy))
        # plt.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()
        plt.savefig("./figs/p4c3_01_n-estimators.png")


    if True:

        tree_counts = [1,2,3,4,5,10,15,20,25,30,40,50, 60, 70, 80, 90, 100, 110, 120, 150]
        # tree_counts = [1,2,3,4,5,10,20, 30,40,50, 75, 100]

        accuracy  = []
        for n_estimator in tree_counts:
            clf = RandomForestClassifier(
                n_estimators = n_estimator,
                max_depth = None,
                max_features = None,
                random_state = 8
                )

            clf.fit(X_train, y_train)
            accuracy.append({
                'n': n_estimator,
                'test': clf.score(X_test, y_test),
                'train': clf.score(X_train, y_train),
            })

            print(n_estimator, clf.score(X_test, y_test), clf.score(X_train, y_train))

        accuracy = pd.DataFrame(accuracy)
        accuracy['delta'] = np.abs(accuracy.train - accuracy.test)

        fig = plt.figure(figsize=(15,6))
        ax = fig.add_subplot(1, 2, 1)
        plt.plot(accuracy.n, accuracy.train, label = 'score train')
        plt.plot(accuracy.n, accuracy.train,'*')

        plt.plot(accuracy.n, accuracy.test, label = 'score test')
        plt.plot(accuracy.n, accuracy.test,'*')

        # plt.plot(accuracy.n, accuracy.delta, label = 'delta')
        # plt.plot(accuracy.n, accuracy.delta,'*')

        ax.grid(True, which = 'both')
        ax.set_title('Accuracy')
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('Accuracy')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_ylim(0.9 * np.min(accuracy), 1.1 * np.max(accuracy))

        ax.legend()
        # --
        ax = fig.add_subplot(1, 2, 2)
        plt.plot(accuracy.n, accuracy.delta, label = 'delta')
        plt.plot(accuracy.n, accuracy.delta,'*')

        ax.grid(True, which = 'both')
        ax.set_title('Différence score(test) - score(train) ')
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('Différence score(test) - score(train)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_ylim(0.9 * np.min(accuracy), 1.1 * np.max(accuracy))

        ax.legend()
        plt.tight_layout()
        plt.show()


        plt.savefig("./figs/p4c3_02_overfit.png")


    if False:
        from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
        from sklearn.model_selection import train_test_split

        filename = './../data/paris-arbres-numerical-2023-09-10.csv'

        data = pd.read_csv(filename)
        X = data[['domanialite', 'arrondissement',
                    'libelle_francais', 'genre', 'espece',
                    'circonference_cm', 'hauteur_m']]
        y = data.stade_de_developpement.values

        X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.8, random_state=808)


        clf = RandomForestClassifier(
            n_estimators = 100,
            random_state = 8
            )

        clf.fit(X_train, y_train)
        print("test :",clf.score(X_test, y_test))
        print('train:', clf.score(X_train, y_train))

        print(clf.feature_importances_)

        df = pd.DataFrame()
        df['feature'] = X.columns
        df['importance'] = clf.feature_importances_
        df.sort_values(by = 'importance', ascending = False, inplace = True)

        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(1, 1, 1)

        sns.barplot(data = df, x='feature', y='importance')
        ax.set_title('Feature importance')
        ax.set_xlabel('Variable')
        ax.set_ylabel('Importance')
        ax.grid(True, which = 'both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

        plt.savefig("./figs/p4c3_03_feature_importance.png")
