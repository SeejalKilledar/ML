import pandas as pd
import numpy as np


df = pd.read_csv('/kaggle/input/digital-recognizer/train.csv')
print(df.head())
df.shape
df.sample()
import matplotlib.pyplot as plt
plt.imshow(df.iloc[19156,1:].values.reshape(28,28))
X = df.iloc[:,1:]
y = df.iloc[:,0]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
X_train.shape
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)
"""
FOR PCA, 
1. SCALING
2. COVARIANCE
3. EIGEN VALUES AND EIGEN VECTOR

"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 200) # if the value is none it will take all 784 cols
X_train_trf = pca.fit_transform(X_train)
X_test_trf = pca.transform(X_test)
knn = KNeighborsClassifier()
knn.fit(X_train_trf, y_train)
y_predict = knn.predict(X_test_trf)
accuracy_score(y_test, y_predict)
for i in range(1,785):
    pca = PCA(n_components = i)
    X_train_trf = pca.fit_transform(X_train)
    X_test_trf = pca.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train_trf, y_train)
    y_predict = knn.predict(X_test_trf)
    print(accuracy_score(y_test, y_predict))

# Visualization
pca = PCA(n_components = 2)
X_train_trf = pca.fit_transform(X_train)
X_test_trf = pca.transform(X_test)

# plot graph via plotly
import plotly.express as px
y_train_trf = y_train.astype(str) # converts int to object
fig = px.scatter(x = X_train_trf[:,0],
                y = X_train_trf[:,1],
                color = y_train_trf,
                color_discrete_sequence=px.colors.qualitative.G10)
fig.show()

# Visualization
pca = PCA(n_components = 3)
X_train_trf = pca.fit_transform(X_train)
X_test_trf = pca.transform(X_test)

y_train_trf = y_train.astype(str) # converts int to object
# fig = px.scatter_3d(x = X_train_trf[:,0],
#                 y = X_train_trf[:,1],
#                  z = X_train_trf[:,2],
#                 color = y_train_trf,
#                 color_discrete_sequence=px.colors.qualitative.G10)

fig = px.scatter_3d(x = X_train_trf[:,0],
                y = X_train_trf[:,1],
                 z = X_train_trf[:,2],
                color = y_train_trf)
fig.update_layout(margin=dict(l=20,r=20,t=20,b=20))

fig.show()

pca.explained_variance_
#Eigen values

pca.components_
# Eigen Vectors

pca.components_.shape

# calculate percentage variance of the original data
pca.explained_variance_ratio_

# the output is giving just 12% of 100% data which is very less


pca = PCA(n_components = None)
X_train_trf = pca.fit_transform(X_train)
X_test_trf = pca.transform(X_test)

plt.plot(np.cumsum(pca.explained_variance_ratio_))