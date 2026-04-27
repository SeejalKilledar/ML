import pandas as pd
import numpy as np

df = pd.read_csv('placement-dataset.csv')
print(df.head())

# df.shape() -> output (100,3)


"""
Steps:
1. Preprocess + EDA + Feature Selection
2. Extract input and output columns
3. Scale the values
4. Train test split
5. Train the model
6. Evaluate the model/Model selection
7. Deploy the model
"""

# 1. Preprocess -> Remove unwanted columns from the file, example : df = df.iloc[:,1:]
# preparing your data for pre-processing

# df = df.iloc[:,1:]
print(df.info())  # check if there is no miss of the data


# EDA -> Exploratory data analysis, creating graphs, get the idea what is hidden in the data by plotting graph
# from the below graph we see there is a line created hence use Logistic Regression
import matplotlib.pyplot as plt
print(plt.scatter(x=df['cgpa'], y = df['iq'], c = df['placement']))


# Feature Selection -> deciding on which column we will be using
# ignoring this as we have a toy data, with less data



# 2. Extract input and output columns -> Seperating input and output columns
input_x = df.iloc[:,0:2]
print(input_x)
print("*********************************")
input_y = df.iloc[:,-1]
print(input_y)
print("********************************")



# 4. Train test split -> also called cross validation , suppose we have 100 data, spli the data into training set and
# testing set, training will have 90data and testing will have 10 data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.1)


# 3. Scale the values -> making the values lie between -1 to 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# y_train = scaler.fit_transform(y_train)
# y_test = scaler.fit_transform(y_test)


# 5. Train the model -> by using training set, we train the model
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
clf = LogisticRegression()
#this is model training
print("Model Training")
clf = make_pipeline(
    SimpleImputer(strategy='mean'),  # or 'median', 'most_frequent', or 'constant'
    LogisticRegression(max_iter=1000)
)
print(clf.fit(x_train, y_train))
x_train = np.where(np.isinf(x_train), np.nan, x_train)
x_train = SimpleImputer(strategy='mean').fit_transform(x_train)

# 6. Evaluate the model/Model selection -> use multiple algo, whichever is performing well use that
#take accuracy
prediction = clf.predict(x_test)
print(y_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction))

# how to see the decision boundry that the alog has created
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x_train, y_train.values, clf=clf, legend=2)



# 7. Deploy the model -> integrate the model with the website, and deploy the website
#create pickle, this will be used to integrate the model

import  pickle
pickle.dump(clf, open('model.pkl', 'wb'))



plt.show()