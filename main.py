# experimenting with data preprocessing
# !python 3

import pandas as pd
import numpy as np
import sklearn.impute as sktemp
import statsmodels.api as sm
import matplotlib.pyplot as plt


def main():
    # import the data
    df = pd.read_csv('adult.csv', na_values=['#NAME?'])

    print(df.head(5))

    # want to look at outcome variable "income"
    print(df['income'].value_counts())

    # assign the outcome as 0 if income is less than 50k, outcome as 1 if more than 50k
    df['income'] = [0 if x == '<=50k' else 1 for x in df['income']]

    # assign X as a dataframe of features and y as a Series of the outcome variables
    X = df.drop('income', 1)
    y = df.income

    print("\n" * 5)

    # see how it looks like
    print(X.head(5))
    print(y.head(5))

    print("\n" * 5)

    # minor data cleaning here
    # education is a categorical feature
    print(X['education'].head(5))

    # use get_dummies in pandas
    print(pd.get_dummies(X['education']).head(5))

    # decide which categorical variables you want to use in the model
    for colName in X.columns:
        if X[colName].dtypes == 'object':
            uniqueCat = len(X[colName].unique())
            print(f"Feature '{colName}' has {uniqueCat} unique categories".format(colName=colName, uniqueCat=uniqueCat))

    # native country has 40 unqiues
    print(X['native_country'].value_counts().sort_values(ascending=False).head(10))

    # bucket the low frequency categories as "Other"
    X['native_country'] = ['United-States ' if x == 'United-States' else 'Other' for x in X['native_country']]
    print(X['native_country'].value_counts().sort_values(ascending=False))

    # Create a list of features to dummy
    todummy_list = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

    # Function to dummy all the categorical variables used for modeling
    def dummy_df(df, todummy_list):
        for x in todummy_list:
            dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
            df = df.drop(x, 1)
            df = pd.concat([df, dummies], axis=1)
        return df

    X = dummy_df(X, todummy_list)
    print(X.head(5))

    print("\n" * 5)

    # handling the missing data
    # How much of your data is missing?
    X.isnull().sum().sort_values(ascending=False).head()

    # Impute missing values using Imputer in sklearn.preprocessing
    imp = sktemp.SimpleImputer(missing_values='NaN', strategy='median', verbose=0)
    imp.fit(X)
    X = pd.DataFrame(data=imp.transform(X), columns=X.columns)

    # Now check again to see if you still have missing data
    X.isnull().sum().sort_values(ascending=False).head()

    print("\n" * 5)

    # outlier detection, Kernal Destiny Estimation
    def find_outliers_kde(x):
        x_scaled = scale(list(map(float, x)))
        kde = sm.nonparametric.KDEUnivariate(x_scaled)
        kde.fit(bw="scott", fft=True)
        pred = kde.evaluate(x_scaled)

        n = sum(pred < 0.05)
        outlier_ind = np.asarray(pred).argsort()[:n]
        outlier_value = np.asarray(x)[outlier_ind]

        return outlier_ind, outlier_value

    kde_indices, kde_values = find_outliers_kde(X['age'])
    print(np.sort(kde_values))

    print("\n" * 5)

    # distribution of features
    # Use pyplot in matplotlib to plot histograms
    def plot_histogram(x):
        plt.hist(x, color='gray', alpha=0.5)
        plt.title("Histogram of '{var_name}'".format(var_name=x.name))
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

    plot_histogram(X['age'])

    # Plot histograms to show distribution of features by outcome categories
    def plot_histogram_dv(x, y):
        plt.hist(list(x[y == 0]), alpha=0.5, label='Outcome=0')
        plt.hist(list(x[y == 1]), alpha=0.5, label='Outcome=1')
        plt.title("Histogram of '{var_name}' by Outcome Category".format(var_name=x.name))
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend(loc='upper right')
        plt.show()

    plot_histogram_dv(X['age'], y)

if __name__ == '__main__':
    main()
