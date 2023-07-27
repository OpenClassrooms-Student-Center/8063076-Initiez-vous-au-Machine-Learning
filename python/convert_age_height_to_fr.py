import pandas as pd
import numpy as np

input_file = "./../data/age_vs_weight_vs_height_vs_gender.csv"
output_file = "./../data/age_vs_poids_vs_taille_vs_sexe.csv"

if  __name__ == "__main__":

    df = pd.read_csv(input_file)

    df.rename(columns = {'sex': 'sexe', 'age': 'age', 'height': 'taille', 'weight': 'poids'}, inplace = True)

    df['taille'] = df.taille.apply(lambda d : np.round(d* 2.54, 2))
    df['poids'] = df.poids.apply(lambda d : np.round(d/ 2.205, 2))
    df.loc[df.sexe == 'f','sexe'] = 1
    df.loc[df.sexe == 'm','sexe'] = 0
    df['sexe'] = df.sexe.astype(int)

    print(df.head())

    df[['sexe', 'age', 'taille', 'poids']].to_csv(output_file, index = False)
