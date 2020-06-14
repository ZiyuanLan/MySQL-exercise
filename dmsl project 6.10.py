
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# read the films data
data = pd.read_csv(r"C:\Users\28940\Desktop\PMA8\PMA_blockbuster_movies")

# remove useless column
data= data.drop(['poster_url','2015_inflation','genres','release_date','title','worldwide_gross'],axis=1)


# feature engineering
film_df = film_df.drop(["Genre_1"], axis = 1)
film_df = film_df.drop(["Genre_2"], axis = 1)
film_df = film_df.drop(["Genre_3"], axis = 1)
film_df = film_df.drop(["rating"], axis = 1)
film_df = film_df.drop(["studio"], axis = 1)

# Separate the target values(Y) from predictors(X)
X = film_df.iloc[:, games_df.columns != "adjusted"]
Y = games_df["adjusted"]

# create the list of scores for testing data
scores_test = {}

# split data into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
		test_size = 0.2, random_state=5)  # X is “1:” and Y is “[0]”

# print the shapes to check everything is OK
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)










