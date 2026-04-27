from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from statsmodels.sandbox.distributions.gof_new import bootstrap

from day61_regression_tree import param_grid

X,y = make_classification(n_samples=10000, n_features=10, n_informative=3)
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(f'Decision tree Accuracy score: {accuracy_score(y_test,y_pred)}')

print("Bagging")
bag = BaggingClassifier(
    # base_estimator = DecisionTreeClassifier(),
    estimator= DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25, #giving 25% of data to 500 estimators
    bootstrap=True, # sampling with replace= True
    random_state=42,
)
bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print(f'Bagging Accuracy score Decision Tree: {accuracy_score(y_test,y_pred)}')

print(bag.estimators_samples_) # this will give the list of indexes in the array
print(bag.estimators_samples_[0].shape) # will give 2000 coz max is 0.25
print(bag.estimators_features_[0].shape) # will return cols shape basicaly all cols coz no col sampling is done

print("Bagging using SVM")
bag = BaggingClassifier(
    estimator= SVC(),
    max_samples= 0.25,
    n_estimators=500,
    bootstrap=True,
    random_state=42
)
bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print(f'Bagging in SVC : {accuracy_score(y_test,y_pred)}')


print("Pasting")
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=False,
    random_state=42,
    verbose=1,
    n_jobs = 1
)

bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print(f"Pasting Decision tree: {accuracy_score(y_test,y_pred)}")

print("Random Subspaces")
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    max_samples= 1.0, # we are considering all the rows
    bootstrap=False,
    n_estimators=500,
    max_features=0.5,
    bootstrap_features=True,
    random_state=42

)
bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print(f"Random Subspaces Decision tree: {accuracy_score(y_test,y_pred)}")
print(bag.estimators_samples_[0].shape)
print(bag.estimators_features_[0].shape)

print("Random Patches")
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    max_features=0.5,
    bootstrap_features=True,
    random_state=42
)

bag.fit(X_train,y_train)
y_pred = bag.predict(X_test)
print(f"Random Patches Decision tree: {accuracy_score(y_test,y_pred)}")
print(bag.estimators_samples_[0].shape)
print(bag.estimators_features_[0].shape)

"""
OOB Score : out of bag sample.
When we do rwo sanpling with replacement, there woul dbe some rows that decisiontree does not get,
37% of the data our bagging algo does not get.
coz of replacement
Use these rows to check the performance of the model
param : oobs_score = True 
"""
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    oob_score=True,
    random_state=42
)
bag.fit(X_train, y_train)
print(bag.oob_score_)
y_pred = bag.predict(X_test)
print(f"OOBS Decision tree: {accuracy_score(y_test,y_pred)}")

print("Applying GridSearchCV")
from sklearn.model_selection import GridSearchCV
parameters = {
    'n_estimators': [50,100,500],
    'n_samples' : [0.1,0.4,0.7,1.0],
    'bootstrap' : [True, False],
    'max_features' : [0.1,0.4,0.7,1.0]
}
search = GridSearchCV(BaggingClassifier(), param_grid, cv = 5)
search.fit(X_train, y_train)
print(search.best_params_)
print(search.best_score_)

