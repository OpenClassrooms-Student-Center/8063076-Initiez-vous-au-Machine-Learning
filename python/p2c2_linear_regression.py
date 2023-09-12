import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

input_file = "./../data/advertising.csv"

if  __name__ == "__main__":

    df = pd.read_csv(input_file)

    df['tv2'] = df.tv**2
    df['tv_radio'] = df.tv * df.radio

    # scale
    if True:

        from sklearn.preprocessing  import MinMaxScaler
        scaler = MinMaxScaler()
        data_array = scaler.fit_transform(df)
        df = pd.DataFrame(data_array, columns = ['tv','radio','journaux','ventes','tv2','tv_radio'])


    reg = LinearRegression()

    # X = df[['tv','radio','journaux']]
    # X = df[['tv','radio','journaux', 'tv2']]
    X = df[['tv','radio','journaux', 'tv_radio', 'tv2']]
    # X = df[['tv','radio','journaux', 'tv_radio']]
    y = df.ventes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    reg.fit(X_train, y_train)
    y_hat_test = reg.predict(X_test)

    print(f"Coefficients :{reg.coef_}")
    print(f"RMSE: {np.round(mean_squared_error(y_test, y_hat_test),  5)}")
    print(f"MAPE: {np.round(mean_absolute_percentage_error(y_test, y_hat_test), 5)}")


    if False:
        fig, ax = plt.subplots(1,3, figsize = (15,5))

        plt.subplot(1,3,1)
        sns.regplot(x = df[['tv']],y  =  df.ventes)
        plt.ylabel('ventes')
        plt.xlabel('TV')
        plt.title('ventes = a * tv  + b')
        plt.grid()
        sns.despine()

        plt.subplot(1,3,2)
        sns.regplot(x = df[['radio']],y  =  df.ventes)
        plt.ylabel('ventes')
        plt.xlabel('radio')
        plt.title('ventes = a * radio + b')
        plt.grid()
        sns.despine()

        plt.subplot(1,3,3)
        res = sns.regplot(x = df[['journaux']],y  =  df.ventes)
        plt.ylabel('ventes')
        plt.xlabel('journaux')
        plt.title('ventes = a * journaux + b')
        plt.grid()
        sns.despine()

        plt.tight_layout()
        plt.show()

        plt.savefig('./../img/advertising_00.png')
