from sklearn.datasets import load_iris
import numpy as np # linear algebra
import pandas as pd



iris = load_iris()
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)
df['Species'] = iris.target

print(df)
# if Species col is categorical, convert it into 0,1,2
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# df['Species'] = encoder.fit_transform(df['Species'])
# df.head()

print(iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# remove setosa and create a df
df = df[df['Species']!=0][['sepal width (cm)', 'petal length (cm)','Species']]
print(df)

import seaborn as sns
import matplotlib.pyplot as plt
plt.scatter(df['sepal width (cm)'], df['petal length (cm)'], c= df['Species'], cmap = 'winter')
plt.show()

#taking 10 rows for training, 5 for testing
df = df.sample(100)
df_train = df.iloc[:60,:].sample(10)
df_val = df.iloc[60:80,:].sample(5)
df_test = df.iloc[80:,:].sample(5)

print(df_train)

# testing on val
x_test = df.iloc[:,0:2].values
y_test = df.iloc[:,-1].values


print("Case1 : Bagging")
df_bag = df_train.sample(8, replace = True)
print(df_bag)
X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score

def evaluate(dt_obj, X,y):
    dt_obj.fit(X,y)
    plot_tree(dt_obj)
    plt.show()
    plot_decision_regions(X.values, y.values,  clf= dt_obj, legend =2)
    y_pred = dt_obj.predict(x_test)
    print(accuracy_score(y_test, y_pred))

df_bag1 = DecisionTreeClassifier()
evaluate(df_bag1,X,y)

df_bag = df_train.sample(8, replace = True)
print(df_bag)
X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]
df_bag2 = DecisionTreeClassifier()
evaluate(df_bag2,X,y)

df_bag = df_train.sample(8, replace = True)
print(df_bag)
X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]
df_bag3 = DecisionTreeClassifier()
evaluate(df_bag3,X,y)

print("Case2 : Aggregation")

print(df_test)
print(f'Prediction 1 : {df_bag1.predict(np.array([2.7,4.2]).reshape(1,2))}')
print(f'Prediction 2 : {df_bag2.predict(np.array([2.7,4.2]).reshape(1,2))}')
print(f'Prediction 3 : {df_bag3.predict(np.array([2.7,4.2]).reshape(1,2))}')

print("Types of Bagging : Pasting, Random Subspace, Random Patches, ")
print("Pasting : Row sampling without replacement")
df_bag = df_train.sample(8, replace = False) # bydefault replace = False
print(df_bag) # no repation of rows if replace = False

print("Random Subspace: col sampling (with or without)")
df_bag = df_train.sample(2, replace = True, axis = 1)
df_bag1 = df_train.sample(2, replace = False, axis = 1)

print("Random patched : col and row sampling both")
df_bag2 = df_train.sample(8, replace = True).sample(2,axis = 1)





