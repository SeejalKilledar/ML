from random import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X,y = make_classification(n_features=5, n_redundant=0, n_informative=5, n_clusters_per_class=1)

df = pd.DataFrame(X, columns=['col1', 'col2', 'col3', 'col4', 'col5'])
df['target'] = y
print(df.shape)
print(df.head())
def sample_rows(df, percent):
    return df.sample(int(percent*df.shape[0]), replace= True)

def sample_features(df, percent):
    cols = random.sample(df.columns.tolist()[:-1],int(percent*df.shape[1]))
    new_df= df[cols]
    new_df['target'] = df['target']
    return new_df

def combined_sampling(df,row_percent, col_percent):
    new_df = sample_rows(df,row_percent)
    return sample_features(new_df, col_percent)

df1 = (sample_rows(df, 0.1))
df2 = (sample_rows(df, 0.1))
df3 = (sample_rows(df, 0.1))

from sklearn.tree import DecisionTreeClassifier
cf1 = DecisionTreeClassifier()
cf2 = DecisionTreeClassifier()
cf3 = DecisionTreeClassifier()

cf1.fit(df1.iloc[:,0:5], df1.iloc[:,-1])
cf2.fit(df2.iloc[:,0:5], df2.iloc[:,-1])
cf3.fit(df3.iloc[:,0:5], df3.iloc[:,-1])

from sklearn.tree import plot_tree
plot_tree(cf1)
plt.show()
plot_tree(cf2)
plt.show()
plot_tree(cf3)
plt.show()

predict1 = cf1.predict(np.array([-0.521621, -0.326016, -2.246965, -1.584046,  2.044984]).reshape(1,5))
print(predict1)
predict2 = cf2.predict(np.array([-0.521621, -0.326016, -2.246965, -1.584046,  2.044984]).reshape(1,5))
print(predict2)
predict3 = cf3.predict(np.array([-0.521621, -0.326016, -2.246965, -1.584046,  2.044984]).reshape(1,5))
print(predict3)