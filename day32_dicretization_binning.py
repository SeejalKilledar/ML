import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer



df = pd.read_csv('titanic.csv', usecols=['Age', 'Fare', 'Survived'])

print(df.shape)
df.dropna(inplace=True)
df.isnull().sum()
print(df.shape)

X = df.iloc[:,1:]
y = df.iloc[:,0]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,
                                                    random_state= 42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

print(np.mean(cross_val_score(clf, X,y, cv = 10, scoring = 'accuracy')))



kbin_age = KBinsDiscretizer(n_bins= 25, encode= 'ordinal', strategy = 'quantile')
kbin_fare = KBinsDiscretizer(n_bins= 25, encode= 'ordinal', strategy = 'quantile')

trf = ColumnTransformer(
    [
        ('first', kbin_age,[0]),
        ('second', kbin_fare,[1])
    ]

)

X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.transform(X_test)

print(trf.named_transformers_['first'].n_bins_)
print(trf.named_transformers_['first'].bin_edges_)

output = pd.DataFrame(
    {
        'age' : X_train['Age'],
        'age_trf':X_train_trf[:,0],
        'fare' : X_train['Fare'],
        'fare_trf':X_train_trf[:,1],

    }

)

print(output)

output['age_labels'] = pd.cut(x=X_train['Age'],
                              bins = trf.named_transformers_['first'].bin_edges_[0].tolist())

output['fare_labels'] = pd.cut(x=X_train['Fare'],
                              bins = trf.named_transformers_['second'].bin_edges_[0].tolist())

clf = DecisionTreeClassifier()
clf.fit(X_train_trf, y_train)

y_pred2 = clf.predict(X_test_trf)

print(accuracy_score(y_test, y_pred2))

print(np.mean(cross_val_score(clf, X,y, cv = 10, scoring = 'accuracy')))

def discretize(bins, strategy):
    kbin_age = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
    kbin_fare = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)

    trf = ColumnTransformer(
        [
            ('first', kbin_age, [0]),
            ('second', kbin_fare, [1])
        ]

    )

    X_trf = trf.fit_transform(X)

    print("Hello")
    print(np.mean(cross_val_score(DecisionTreeClassifier(), X, y, cv=10, scoring='accuracy')))


    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.hist(X['Age'])
    plt.title("Before")

    plt.subplot(122)
    plt.hist(X_trf[:,0], color = 'blue')
    plt.title("After")

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.hist(X['Fare'])
    plt.title("Before")

    plt.subplot(122)
    plt.hist(X_trf[:,1], color='red')
    plt.title("After")

    plt.show()

discretize(10, 'uniform')