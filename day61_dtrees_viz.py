import graphviz.backend as be
import numpy as np
from dtreeviz import dtreeviz, explain_prediction_path, rtreeviz_univar, rtreeviz_bivar_3D
import dtreeviz
from pyexpat import features

from sklearn.datasets import *
from dtreeviz.trees import *
from IPython.display import Image,display_svg, SVG
from testing.animate_rtree_bivar_3D import figsize

clas = tree.DecisionTreeClassifier(max_depth=1)
iris = load_iris()
X_train = iris.data
#print(X_train)
y_train = iris.target
clas.fit(X_train, y_train)

from sklearn.tree import plot_tree
plot_tree(clas)
#plt.show()

print("***** Classification *****")
# m = dtreeviz(clas, X_train, y_train, feature_names = iris.feature_names,
#                    class_names = ['setosa', 'versicolor','virginica'],x=X_train[0],
#                    scale = 1.5)

m = dtreeviz.model(
    clas,
    X_train,
    y_train,
    feature_names=iris.feature_names,
    class_names=['setosa', 'versicolor','virginica']
)

viz = m.view(x=X_train[0])
print(viz)

print("2. Regression")
regr = tree.DecisionTreeClassifier(max_depth=2)
boston = load_diabetes()
X_train = boston.data
print(boston.feature_names)
y_train = boston.target
regr.fit(X_train, y_train)

viz = dtreeviz.model(regr,
                     X_train, y_train,
                     target_name='s6',
                     feature_names=boston.feature_names)

viz = m.view(x=X_train[0])
viz.save("boston_tree.svg")

print("Horizontal Decision Tree")

viz = dtreeviz.model(clas,
                     X_train, y_train,
                     feature_names = iris.feature_names,
                     class_names = ['setosa', 'versicolor', 'virginica'],
                     scale = 2,
                     orientation = 'LR')


print("Show prediction path")
clas = tree.DecisionTreeClassifier()
iris = load_iris()
X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)

# take any flower/row and plot its prediction path
X = iris.data[np.random.randint(0, len(iris.data)),:]
viz = dtreeviz.model(
    clas,
    X_train, y_train,
    feature_names = iris.feature_names,
    class_names = ['setosa', 'versicolor', 'virginica'],
    X=X
)

# show node number
print("********* Show node number ********* ")
viz = dtreeviz.model(
    clas,
    X_train, y_train,
    feature_names= iris.feature_names,
    class_names=['setosa','versicolor', 'virginica'],
    histtype = 'barstacked',
    scale = 1.5,
    orientation = 'LR',
    show_node_labels = True,
)

print("****** Without any graphs ********")
viz = dtreeviz.model(
    clas,
    X_train, y_train,
    feature_names= iris.feature_names,
    class_names=['setosa','versicolor', 'virginica'],
    histtype = 'barstacked',
    scale = 1.5,
    orientation = 'LR',
    show_node_labels = True,
    fancy = False
)

print("Prediction in English")
print(explain_prediction_path(clas, X, feature_names=iris.feature_names, explanation_type='plain_english'))

print("Feature Importance")
print(explain_prediction_path(clas, X, feature_names=iris.feature_names, explanation_type='plain_english' ))

print("Univariate Regression")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from dtreeviz import dtreeviz
import  dtreeviz


df_cars = pd.read_csv('Cars dataset.csv')
X, y = df_cars[['WGT']], df_cars[['MPG']]
dt = DecisionTreeClassifier(criterion='mse', max_depth=3)
dt.fit(X,y)
fig = plt.figure()
ax = fig.gca()
rtreeviz_univar(dt,X,y,'WGT', 'MPG', ax = ax)
plt.show()

print("3D - Regression")
from mpl_toolkits.mplot3d import axes3d

X = df_cars[['WGT', 'ENG']]
y = df_cars['MPG']
dt = DecisionTreeClassifier(criterion='mae', max_depth=3)
dt.fit(X,y)

figsize = (6,5)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection = '3d')

t = rtreeviz_bivar_3D(dt, X,y, feature_names = ['vehical Weight', 'Horse Power'],
                        target_name = 'MPG', fontsize = 14, elev = 20, azim = 25, dist = 8.2,
                        show={'splits', 'title'}, ax=ax)

plt.show()