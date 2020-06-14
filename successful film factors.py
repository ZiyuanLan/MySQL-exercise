
# ZIYUAN LAN 1958690
# Film project Code

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LogisticRegression as LREG
from sklearn.svm import SVR

# read the films data
data = pd.read_csv('PMA_blockbuster_movies.csv')

#  Data cleaning
#  missing values processing
def missing_values_table(df):
    # Check all missing values
    mis_val = df.isnull().sum()

    # Proportion of missing values
    mis_val_percent = 100 * mis_val / len(df)

    # Create a table containing the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Set column names
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Number of Missing Values', 1: 'Percentage of Total Values (%)'})

    # Sort the table by percentage of missing values (descending order)
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        'Percentage of Total Values (%)', ascending=False).round(3)

    # Print summary information
    print("The dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns having missing values.")

    # Results output
    return mis_val_table_ren_columns

# delete null rows
data = data[data['title'].notnull()]

# drop unused columns
data = data.drop(['poster_url','title','Genre_1','Genre_2','Genre_3','year'], axis=1)

# get dummies on column 'genres'
genres = data['genres'].str.get_dummies('\n')
data = pd.concat([data, genres], axis='columns')
data = data.drop(['genres'],axis=1)

# get dummies on column 'rating'
data = pd.get_dummies(data, columns=['rating'])

# data type conversion
data['2015_inflation'] = data['2015_inflation'].replace('\%','', regex=True).astype('float64')
data['adjusted'] = data['adjusted'].replace('\$','', regex=True).replace('\,','', regex=True).astype('float64')
data['worldwide_gross'] = data['worldwide_gross'].replace('\$','', regex=True).replace('\,','', regex=True).astype('float64')

# get dummies on column 'release_date'
# date离散化成0-1向量(比如月级别为12维向量)
data['release_date'] = data['release_date'].apply(lambda row : pd.to_datetime(row))
data['month'] = data['release_date'].apply(lambda d: d.month)
data['day'] = data['release_date'].apply(lambda d: d.day)
data['year'] = data['release_date'].apply(lambda d: d.year)
data = data.drop(['release_date'],axis=1)
data = pd.get_dummies(data, columns=['month','day','year'])

# get dummies on column 'studio'
studio = data['studio'].str.get_dummies(' / ')
data = pd.concat([data, studio], axis='columns')
data = data.drop(['studio'],axis=1)

# print(data)
# print(data.dtypes)
# data.to_excel('test.xlsx')

label = data['adjusted'].astype('int64')
data = data.drop(['adjusted'],axis=1)

# Normalization
data = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)

# split training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size = 0.1, random_state=5)

# print the shapes to check everything is OK
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

def dtr_model():
    print("===================================DecisionTreeRegressor model start===================================")
    # Set the parameters
    tuned_parameters = [{'criterion': ['mse','mae'],
                         'max_depth': np.arange(1,10)}]
    print("# Tuning hyperparameters:")
    print("\n")
    dtr = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=5)
    dtr.fit(X_train, Y_train)

    print("Best parameters set found on the training set:")
    print(dtr.best_params_)
    print("\n")

    print("Mean test score on the training set:")
    print(dtr.cv_results_['mean_test_score'])
    print("\n")

    # print("Std test score on the training set:")
    # print(rfr.cv_results_['std_test_score'])
    # print("\n")

    print("Best score on the training set:")
    print(dtr.best_score_)
    print("\n")

    print("Score on the testing set:")
    print(dtr.score(X_test, Y_test))
    print("\n")

    print("Predicted box office on the testing set:")
    predicted = dtr.predict(X_test)
    print(predicted)

    train_rmse = sqrt(mean_squared_error(dtr.predict(X_train), Y_train))
    print('Train RMSE: %.3f' % train_rmse)

    test_rmse = sqrt(mean_squared_error(predicted, Y_test))
    print('Test RMSE: %.3f' % test_rmse)
    print("===================================DecisionTreeRegressor model end===================================\n")

def rfr_model():
    print("===================================RandomForestRegressor model start===================================")
    # Set the parameters
    tuned_parameters = {'min_samples_split':[3,6,9],
                        'n_estimators':[10,50,100]}
    # min_samples_split:分裂内部节点需要的最少样例数.int(具体数目),float(数目的百分比)
    # n_estimators:森林中数的个数。这个属性是典型的模型表现与模型效率成反比的影响因子,即便如此,你还是应该尽可能提高这个数字,以让你的模型更准确更稳定。
    print("Tuning hyperparameters:")
    print("\n")
    rfr = GridSearchCV(RandomForestRegressor(), param_grid=tuned_parameters, cv=5)
    rfr.fit(X_train, Y_train)

    print("Best parameters set found on the training set:")
    print(rfr.best_params_)
    print("\n")

    print("Mean test score on the training set:")
    print(rfr.cv_results_['mean_test_score'])
    print("\n")

    # print("Std test score on the training set:")
    # print(rfr.cv_results_['std_test_score'])
    # print("\n")

    print("Best score on the training set:")
    print(rfr.best_score_)
    print("\n")

    print("Score on the testing set:")
    print(rfr.score(X_test, Y_test))
    print("\n")

    print("Predicted box office on the testing set:")
    predicted = rfr.predict(X_test)
    print(predicted)

    train_rmse = sqrt(mean_squared_error(rfr.predict(X_train), Y_train))
    print('Train RMSE: %.3f' % train_rmse)

    test_rmse = sqrt(mean_squared_error(predicted, Y_test))
    print('Test RMSE: %.3f' % test_rmse)
    print("===================================RandomForestRegressor model end===================================\n")

def voting():
    # dtr model
    tuned_parameters = [{'criterion': ['mse','mae'],
                             'max_depth': np.arange(1,10)}]
    dtr = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=5)

    # rfr model
    tuned_parameters = {'min_samples_split': [3, 6, 9],
                        'n_estimators': [10, 50, 100]}
    rfr = GridSearchCV(RandomForestRegressor(), param_grid=tuned_parameters, cv=5)

    # build voting model
    voting_reg = VotingRegressor(estimators=[
        ('dtr_reg', dtr),
        ('rfr_reg', rfr)
    ],weights=[1,2])

    # fit the model using some training data
    voting_reg.fit(X_train, Y_train)

    # print the mean accuracy of testing predictions
    train_score = voting_reg.score(X_test, Y_test)

    # print the mean accuracy of testing predictions
    print("Accuracy score for final voting= " + str(round(train_score, 4)))

dtr_model()
rfr_model()
voting()

