import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.tree import plot_tree
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

X, y = make_classification(n_features=5, n_informative=5, n_redundant=0, n_clusters_per_class=1)
df = pd.DataFrame(X, columns=['col1', 'col2', 'col3', 'col4', 'col5'])
df['target'] = y
print(X.shape)
print(df.head())

bag = BaggingClassifier(max_features=2)
bag.fit(df.iloc[:,0:5], df.iloc[:,-1])

plt.figure(figsize=(18,8))
plot_tree(bag.estimators_[0])
plt.show()

rf = RandomForestClassifier(max_features=2)
rf.fit(df.iloc[:,0:5], df.iloc[:,-1])

plt.figure(figsize=(18,8))
plot_tree(rf.estimators_[0])
plt.show()