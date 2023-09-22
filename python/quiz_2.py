# Quiz 2

# load from original and cleanup
if True:
    # pip install palmerpenguins
    from palmerpenguins import load_penguins
    data = load_penguins()
    data = data.sample(frac = 1, random_state = 808)
    data.dropna(inplace = True)
    data.reset_index(inplace = True, drop = True)
    data.to_csv("./../data/palmer_penguins_openclassrooms.csv", index = False)

else:

    filename = "https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/palmer_penguins_openclassrooms.csv"
    data = pd.read_csv(filename)

# Q4
if True:
    scaler = MinMaxScaler()
    y = data['body_mass_g']
    X = scaler.fit_transform(data[['bill_length_mm','bill_depth_mm','flipper_length_mm']])
    reg = LinearRegression()

    import seaborn as sns
    for test_size in [0.2, 0.5, 0.8]:
        score = []
        for random_state in np.arange(200):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            reg.fit(X_train, y_train)
            score.append(reg.score(X_test, y_test))

        fig = plt.figure(figsize=(6, 6))
        sns.boxplot(score)
        plt.title(f"test_size {test_size}")
        plt.show()
