import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats # plotting QQ Plot
from seaborn import kdeplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer



df = pd.read_csv('titanic.csv', usecols= ['Age', 'Fare', 'Survived'])
print(df.isnull().sum())

# fill age and fare with mean as there are null values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
print(df.isnull().sum())


# train and test split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis = 1),
                                                    df['Survived'], test_size= 0.2,
                                                    random_state= 0
                                                    )
print("Hello ************")
print(X_train.head())
print(X_train.shape)
print(X_test.shape)

"""



"""



"""
Plot distplot and QQ plot for Age and Fare to check the distribution
"""

plt.figure(figsize=(14,4))
plt.subplot(121)
sns.displot(X_train['Age'], kde=True)
plt.title("Age PDF")

plt.subplot(122)
stats.probplot(X_train['Age'], dist = "norm", plot = plt)
plt.title("Age QQ Plot")

# Fare
plt.figure(figsize=(14,4))
plt.subplot(121)
sns.displot(X_train['Fare'], kde=True)
plt.title("Fare PDF")

plt.subplot(122)
stats.probplot(X_train['Fare'], dist = "norm", plot = plt)
plt.title("Fare QQ Plot")


clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train, y_train)
clf2.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred1 = clf2.predict(X_test)

print(f"Accuracy LR {accuracy_score(y_test, y_pred)}")
print(f"Accuracy DT {accuracy_score(y_test, y_pred1)}")

print("***********************************************************")
print("**** Applying Log Transformer to check if the above accuracy score improves")

trf = FunctionTransformer(func = np.log1p)
X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.fit_transform(X_test)

clf2 = LogisticRegression()
clf3 = DecisionTreeClassifier()

clf2.fit(X_train_transformed, y_train)
clf3.fit(X_train_transformed, y_train)


y_pred2 = clf2.predict(X_test_transformed)
y_pred3 = clf3.predict(X_test_transformed)

print(f'Accuracy LR : {accuracy_score(y_test, y_pred2)}')
print(f'Accuracy DT : {accuracy_score(y_test, y_pred3)}')

print("******* Perform Cross validation to check the accuracy score *****")
X = df.iloc[:,1:3]
y = df.iloc[:,0]
X_transform = trf.fit_transform(X)
clf4 = LogisticRegression()
clf5 = DecisionTreeClassifier()
print(np.mean(cross_val_score(clf4, X_transform,y, scoring='accuracy', cv = 10)))
print(np.mean(cross_val_score(clf5, X_transform,y, scoring='accuracy', cv = 10)))



# function to chekc other transforms, square, squareroot

def apply_transform(transform):
    X = df.iloc[:,1:3]
    y = df.iloc[:,0]

    trf = ColumnTransformer(

        [
            ('log', FunctionTransformer(transform),['Fare'])
        ], remainder= 'passthrough'
    )

    X_trans = trf.fit_transform(X)

    clf = LogisticRegression()

    print(np.mean(cross_val_score(clf, X_trans, y, scoring='accuracy', cv=10)))

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    stats.probplot(X['Fare'], dist="norm", plot=plt)
    plt.title("Fare Before Transformation")

    plt.subplot(122)
    stats.probplot(X_trans[:,0], dist="norm", plot=plt)
    plt.title("Fare QQ Plot")

    plt.show()

apply_transform(lambda x: x**2)



#plt.show()