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

from category_encoders.ordinal import OrdinalEncoder

if __name__ == "__main__":

    filename = './../data/paris-arbres-2023-09-07.csv'

    data = pd.read_csv(filename, sep = ';')

    data.columns = [ col.lower().replace(' / ','_').replace(' ','_').replace('(','').replace(')','')       for col in  data.columns]

    df = data[data.libelle_francais == 'Platane'].copy()

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
    df['stade_de_developpement'] = stade.copy()
    df['stade_de_developpement'].value_counts()
