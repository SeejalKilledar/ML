import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('mnist.csv')
print(df.head())
X = df.iloc[:,1:]
y = df.iloc[:,0]
sns.heatmap(X.iloc[5].values.reshape(28,28))

rf = RandomForestClassifier()
rf.fit(X,y)
print(rf.feature_importances_)
print(rf.feature_importances_.shape)
sns.heatmap(rf.feature_importances_.reshape(28,28))
