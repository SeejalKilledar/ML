import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


df = pd.read_csv('titanic_toy.csv')
print(df.head())

print(df.isnull().mean())

X = df.drop(columns='Survived')
y = df['Survived']

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size = 0.2,
                                                   random_state=2)

print(X_train.shape, X_test.shape)

print(X_train.isnull().mean())

X_train['Age_99'] = X_train['Age'].fillna(99)
X_train['Age_minus1'] = X_train['Age'].fillna(-1)

X_train['Fare_999'] = X_train['Fare'].fillna(999)
X_train['Fare_minus1'] = X_train['Fare'].fillna(-1)

print(X_train)

print(f'Original age variable Variance {X_train["Age"].var()}')
print(f'Age variance after adding 99 {X_train["Age_99"].var()}')
print(f'Age variance after adding -1 {X_train["Age_minus1"].var()}')

print(f'Original Fare variable Variance {X_train["Fare"].var()}')
print(f'Fare variance after adding 99 {X_train["Fare_999"].var()}')
print(f'Fare variance after adding -1 {X_train["Fare_minus1"].var()}')


# plotting graph

fig = plt.figure()
ax = fig.add_subplot(111)
X_train['Age'].plot(kind = 'kde', ax = ax, color = 'red')
X_train['Age_99'].plot(kind = 'kde', ax = ax, color = 'green')
X_train['Age_minus1'].plot(kind = 'kde', ax = ax, color = 'blue')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc = 'best')

fig = plt.figure()
ax1 = fig.add_subplot(111)
X_train['Fare'].plot(kind = 'kde', ax = ax1, color = 'red')
X_train['Fare_999'].plot(kind = 'kde', ax = ax1, color = 'green')
X_train['Fare_minus1'].plot(kind = 'kde', ax = ax1, color = 'blue')
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc = 'best')

plt.show()