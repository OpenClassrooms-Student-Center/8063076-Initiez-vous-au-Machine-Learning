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

if __name__ == "__main__":

    filename = './../data/paris-arbres-2023-09-07.csv'

    data = pd.read_csv(filename, sep = ';')

    data.columns = [ col.lower().replace(' / ','_').replace(' ','_').replace('(','').replace(')','')       for col in  data.columns]

    data = data[data.remarquable != 'OUI'].copy()

    unknown = data[data.stade_de_developpement.isna()].copy()
    data.dropna(subset = ['stade_de_developpement'], inplace = True)

    data = data[(data.circonference_cm > 0) & (data.hauteur_m > 0)].copy()

    columns = ['domanialite', 'arrondissement',
        'libelle_francais', 'genre', 'espece',
        'circonference_cm', 'hauteur_m', 'stade_de_developpement']

    data = data[columns]
    data = data.sample(frac = 1, random_state = 808)
    data.reset_index(inplace = True, drop = True)

    categorical = ['domanialite', 'arrondissement', 'libelle_francais', 'genre', 'espece']
    target = 'stade_de_developpement'
    encoder = OrdinalEncoder(cols = categorical)
    numeric = encoder.fit_transform(data[categorical])
    X = pd.concat([numeric[categorical], data[['circonference_cm', 'hauteur_m']]], axis = 1)

    mapping =[ {'col': 'stade_de_developpement',
        'mapping': {
                    np.nan: 0,
                    'Jeune (arbre)': 1,
                    'Jeune (arbre)Adulte': 2,
                    'Adulte': 3,
                    'Mature': 4
                    }
                } ]

    if True:
        target_encoder = OrdinalEncoder(mapping = mapping)
        y = target_encoder.fit_transform(data[target]).stade_de_developpement.values

    # if True:
    #     stades = ['Jeune (arbre)', 'Jeune (arbre)Adulte', 'Adulte', 'Mature']
    #     data['target'] = 0
    #     data.loc[data.stade_de_developpement.isin(['Jeune (arbre)'])  ,'target'] = 1
    #     y = data.target.values

    ## -- sauver les data

    # ---
    from sklearn.linear_model import LogisticRegression

    # ---
    # from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, random_state=0
    )


    # for loss in ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']:
    # loss = 'log_loss'
    # for loss in ['log_loss']:

    score = []
    # for n_samples in np.arange(1000, X_train_all.shape[0], 10000):
    #     X_train = X_train_all[:n_samples].copy()
    #     y_train = y_train_all[:n_samples].copy()
    # print("=="*10, n_samples)
    # clf = make_pipeline(
    #             StandardScaler(),
    #
    #             SGDClassifier( loss =loss,
    #                 # eta0 = 0.001,
    #                 learning_rate = 'optimal',
    #                 penalty = None,
    #                 random_state = 808,
    #             )
    #     )
    scores = []
    for md in np.arange(2,  30, 2):
        clf = DecisionTreeClassifier(
            max_depth = md,
            random_state = 808
        )

        clf.fit(X_train, y_train)
        if False:
            print("-- score")
            print("train", clf.score(X_train, y_train))
            print("test", clf.score(X_test, y_test))

        y_train_hat = clf.predict(X_train)
        y_test_hat = clf.predict(X_test)

        try:
            print('-- roc auc')
            # train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
            # test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            train_auc = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
            test_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
            scores.append({
                'max_depth': md,
                # 'train': clf.score(X_train, y_train),
                # 'test': clf.score(X_test, y_test),
                'train': train_auc,
                'test': test_auc,
            })

            print("train",train_auc)
            print("test", test_auc)
            ok = True
        except:
            ok = False
            pass

        if False:
            print('-- confusion matrix')
            print(confusion_matrix(y_test, y_test_hat))

            print('-- classification report')
            print(classification_report(y_test, y_test_hat))


        if False & ok:
            fig = plt.figure(figsize=(9,9))
            for k in range(4):
                ax = fig.add_subplot(2, 2, k + 1)
                plt.hist(clf.predict_proba(X_test)[:,k], bins = 100)
                ax.grid()
                ax.set_title(k)
            plt.show()

    scores = pd.DataFrame(scores)


    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(scores.max_depth, scores.train, label = 'train')
    plt.plot(scores.max_depth, scores.test, label = 'test')
    plt.plot(scores.max_depth, scores.train, '*')
    plt.plot(scores.max_depth, scores.test, 'o')
    ax.grid(True, which = 'both')
    ax.set_title('Train and test AUC vs max_depth')
    ax.set_xlabel('Max depth')
    ax.set_ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig("./figs/p3c3_02_overfit.png")
