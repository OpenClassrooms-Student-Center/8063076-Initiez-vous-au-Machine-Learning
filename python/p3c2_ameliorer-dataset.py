import streamlit as st
import openai
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

if __name__ == "__main__":

    filename = './../data/paris-arbres-2023-09-07.csv'

    data = pd.read_csv(filename, sep = ';')

    data.columns = [ col.lower().replace(' / ','_').replace(' ','_').replace('(','').replace(')','')       for col in  data.columns]

    df = data[data.libelle_francais == 'Platane'].copy()

    if False:
        # outliers
        df = data[data.libelle_francais == 'Platane'].copy()

        fig = plt.figure(figsize=(6, 6))
        plt.plot(df.circonference_cm, df.hauteur_m, '.')
        plt.grid()
        plt.xlabel('circonférence (cm)')
        plt.ylabel('hauteur (m)')
        plt.title('Platanes')
        plt.show()
        plt.savefig("./figs/p3c2_01_outliers.png")

    if False:
        # missing values
        df = df[(df.circonference_cm < 400) & (df.hauteur_m < 40)].copy()

        df.stade_de_developpement.value_counts(dropna = False)
        cats = ['Jeune (arbre)', 'Jeune (arbre)Adulte', 'Adulte', 'Mature']

        for n, cat in zip(range(4), cats):
            df.loc[df.stade_de_developpement == cat, 'order_'] = n
        df.sort_values(by = 'order_', inplace = True)


        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(1, 2, 1)
        sns.boxplot(data = df, y="circonference_cm", x="stade_de_developpement")
        ax.grid(True, which = 'both')
        ax.set_title('distribution de la circonférence par stade de développement')

        ax = fig.add_subplot(1, 2, 2)
        sns.boxplot(data = df, y="hauteur_m", x="stade_de_developpement")
        ax.grid(True, which = 'both')
        ax.set_title('distribution de la hauteur par stade de développement')

        plt.tight_layout()
        plt.savefig("./figs/p3c2_03_stade-developpement.png")
        # jeunes
        cond = (df.stade_de_developpement.isna()) & (df.hauteur_m < 8) & (df.circonference_cm < 50)
        df[cond].shape
        # Mature
        cond = (df.stade_de_developpement.isna()) & (df.hauteur_m > 20) & (df.circonference_cm > 200)
        df[cond].shape


    if False:
        df = df[(df.circonference_cm < 1000) & (df.hauteur_m < 100)].copy()

        from scipy import stats
        df['z_circonference'] = stats.zscore(df.circonference_cm)
        df['z_hauteur'] = stats.zscore(df.hauteur_m)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        df.z_circonference.hist(bins = 100)
        ax.grid(True, which = 'both')
        ax.set_title('circonférence - z-score')
        ax = fig.add_subplot(1, 2, 2)
        df.z_hauteur.hist(bins = 100)
        ax.grid(True, which = 'both')
        ax.set_title('hauteur - z-score')
        plt.show()
        plt.savefig("./figs/p3c2_02_zscore.png")


        df[df.z_hauteur> 2].shape
        df[df.z_hauteur> 3].shape
        df[df.hauteur_m > upper].shape
        df[df.z_circonference> 2].shape
        df[df.z_circonference> 3].shape
        df[df.circonference_cm > upper].shape

        iqr = np.quantile(df.hauteur_m, q=[0.25, 0.75])
        limite_basse =  iqr[0] - 1.5*(iqr[1] - iqr[0])
        limite_haute =  iqr[1] + 1.5*(iqr[1] - iqr[0])

    if False:
        df = df[(df.circonference_cm < 1000) & (df.hauteur_m < 100)].copy()
        pd.qcut(df.hauteur_m, 3, labels=["petit", "moyen", "grand"]).value_counts()

        # ordre de grandeur
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        plt.hist(df.circonference_cm, bins = 100)
        ax.grid(True, which = 'both')
        ax.set_title('circonférence')

        ax = fig.add_subplot(1, 2, 2)
        plt.hist(df.hauteur_m, bins = 100)
        ax.grid(True, which = 'both')
        ax.set_title('hauteur')
        plt.show()


        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df['circonference_standard'] = scaler.fit_transform(df.circonference_cm.values.reshape(-1, 1))
        df['hauteur_standard'] = scaler.fit_transform(df.hauteur_m.values.reshape(-1, 1))

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        plt.hist(df.circonference_standard, bins = 100)
        ax.grid(True, which = 'both')
        ax.set_title('circonférence')

        ax = fig.add_subplot(1, 2, 2)
        plt.hist(df.hauteur_standard, bins = 100)
        ax.grid(True, which = 'both')
        ax.set_title('hauteur')
        plt.show()

        from sklearn.preprocessing import MinMaxScaler
        # Create a MinMaxScaler instance
        scaler = MinMaxScaler()
        df['circonference_norm'] = scaler.fit_transform(df.circonference_cm.values.reshape(-1, 1))
        df['hauteur_norm'] = scaler.fit_transform(df.hauteur_m.values.reshape(-1, 1))

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        plt.hist(df.circonference_norm, bins = 100)
        ax.grid(True, which = 'both')
        ax.set_title('circonférence')

        ax = fig.add_subplot(1, 2, 2)
        plt.hist(df.hauteur_norm, bins = 100)
        ax.grid(True, which = 'both')
        ax.set_title('hauteur')
        plt.show()

        # log
        df['circonference_log'] = np.log(df.circonference_cm + 1)
        df['hauteur_log'] = np.log(df.hauteur_m + 1)


        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        plt.hist(df.circonference_log, bins = 100)
        ax.grid(True, which = 'both')
        ax.set_title('histogramme log(circonférence)')

        ax = fig.add_subplot(1, 2, 2)
        plt.hist(df.hauteur_log, bins = 100)
        ax.grid(True, which = 'both')
        ax.set_title('histogramme log(hauteur)')
        plt.show()

        plt.tight_layout()

        plt.savefig("./figs/p3c2_03_log.png")


    if True:
        # figure  recapitulative
        dataset_url = "https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/paris-arbres-clean-2023-09-10.csv"
        data = pd.read_csv(dataset_url)
        df = data[data.libelle_francais == 'Platane'].copy()

        df = df[~df.stade_de_developpement.isna() & (df.circonference_cm !=  0) & (df.hauteur_m != 0) & (df.circonference_cm < 1000) & (df.hauteur_m < 100)].copy()
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        minmax_scaler = MinMaxScaler()
        df['hauteur_minmax'] = minmax_scaler.fit_transform(df.hauteur_m.values.reshape(-1, 1))
        df['circonference_minmax'] = minmax_scaler.fit_transform(df.circonference_cm.values.reshape(-1, 1))

        # Z-score
        standard_scaler = StandardScaler()
        df['hauteur_standard'] = standard_scaler.fit_transform(df.hauteur_m.values.reshape(-1, 1))
        df['circonference_standard'] = standard_scaler.fit_transform(df.circonference_cm.values.reshape(-1, 1))

        # log
        df['circonference_log'] = np.log(df.circonference_cm + 1)
        df['hauteur_log'] = np.log(df.hauteur_m + 1)



        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(3,2, 1)
        ax.hist(df.hauteur_minmax, bins = 100)
        ax.set_title('Hauteur MinMax')
        ax.grid()

        ax = fig.add_subplot(3,2, 2)
        ax.hist(df.circonference_minmax, bins = 100)
        ax.set_title('Circonférence MinMax')
        ax.grid()

        ax = fig.add_subplot(3,2, 3)
        ax.hist(df.hauteur_standard, bins = 100)
        ax.set_title('Hauteur Z-score')
        ax.grid()

        ax = fig.add_subplot(3,2, 4)
        ax.hist(df.circonference_standard, bins = 100)
        ax.set_title('Circonférence Z-score')
        ax.grid()

        ax = fig.add_subplot(3,2, 5)
        ax.hist(df.hauteur_log, bins = 100)
        ax.set_title('Hauteur log')
        ax.grid()

        ax = fig.add_subplot(3,2, 6)
        ax.hist(df.circonference_log, bins = 100)
        ax.set_title('Circonférence log')
        ax.grid()

        plt.tight_layout()
        plt.show()
        plt.savefig("./figs/p3c2_04_recap.png")
