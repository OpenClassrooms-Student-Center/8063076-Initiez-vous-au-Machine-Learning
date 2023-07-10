
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


filename  = "../data/age_vs_weight_vs_height_vs_gender.csv"

if __name__ == "__main__":

    df = pd.read_csv(filename)


    sns.lmplot(
        data=df,
        x="age",
        y="weight",
        hue="sex",
        palette="muted",
        ci=None,
    )
    plt.title("Poids vs Âge")
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig("figs/p1c2_01_a.png")


    sns.lmplot(
        data=df,
        x="height",
        y="weight",
        hue="sex",
        palette="muted",
        ci=None,
    )
    plt.title("Poids vs Taille")
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig("figs/p1c2_01_b.png")
