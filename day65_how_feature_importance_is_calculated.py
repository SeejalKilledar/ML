import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

X,y = make_classification(n_samples=5, n_classes=2, n_features=2, n_informative=2,n_redundant=0, random_state=0)
clf = DecisionTreeClassifier()
clf.fit(X,y)
plot_tree(clf)
plt.show()
print(clf.feature_importances_)
rf = RandomForestClassifier(n_estimators=2)
rf.fit(X,y)
print(rf.feature_importances_)
print(rf.estimators_[0].feature_importances_)
print(rf.estimators_[1].feature_importances_)
print((rf.estimators_[0][0].feature_importances_ + rf.estimators_[1][0].feature_importances_)/2)
print((rf.estimators_[1][0].feature_importances_ + rf.estimators_[1][1].feature_importances_)/2)

print("Multiple nodes")
X,y = make_classification(n_samples=15, n_classes=2, n_features=2, n_informative=2,n_redundant=0, random_state=0)
clf = DecisionTreeClassifier()
clf.fit(X,y)
plot_tree(clf)
plt.show()
print(clf.feature_importances_)