#sup_func
def popu_cifra(x):
    if x >= 1e9:
        s = '{:1.2f}B'.format(x*1e-9)
    elif x >= 1e6:
        s = '{:1.1f}M'.format(x*1e-6)
    else:
        s = '{:1.0f}K'.format(x*1e-3)
    return s

def get_cat_popu(list_populations):
    dic_poblac = {}
    for population in list_populations:
        if population>1e9:
            dic_poblac[population] = 1
        elif population>1e8:
            dic_poblac[population] = 2
        elif population>3e7:
            dic_poblac[population] = 3
        elif population>1e7:
            dic_poblac[population] = 4
        elif population>1e6:
            dic_poblac[population] = 5
        else:
            dic_poblac[population] = 6
    return dic_poblac

def edit_df(df, resp):

    if resp == 'TotalCases':
        df = df.drop(['TotalDeaths','TotalRecovered'],axis=1)
    elif resp == 'TotalDeaths':
        df.drop(['TotalCases','TotalRecovered'],axis=1, inplace=True)
    elif resp == 'TotalRecovered':
        df.drop(['TotalCases','TotalDeaths'],axis=1, inplace=True)
    else:
        print('error, no right input')
    df = df.dropna(subset=[resp], axis=0)
    df['Migrants (net)'].fillna(0, inplace=True)
    df['TotalTests'].fillna(0, inplace=True)
    df['Med. Age'].fillna((df['Med. Age'].mean()), inplace=True)
    df['Urban Pop %'].fillna((df['Urban Pop %'].mean()), inplace=True)
    y= df[resp]
    df = df.drop([resp], axis=1)
    X = df

    return X, y

def get_stat_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

#Rsquared and y_test
    rsquared_score = r2_score(y_test, y_test_preds)
    length_y_test = len(y_test)
    length_y_train = len(y_train)
    test_score= r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)
    return test_score, train_score, y_test, y_train



#Use the function to create X and y


def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test
