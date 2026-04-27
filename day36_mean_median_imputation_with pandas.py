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

mean_age = df['Age'].mean()
median_age = df['Age'].median()

mean_fare = df['Fare'].mean()
median_fare = df['Fare'].median()

X_train['Age_mean'] = X_train['Age'].fillna(mean_age)
X_train['Age_median'] = X_train['Age'].fillna(median_age)

X_train['Fare_mean'] = X_train['Fare'].fillna(mean_fare)
X_train['Fare_median'] = X_train['Fare'].fillna(median_fare)

print(X_train)

print(X_train.isnull().mean())


# Calculate Variance between original age and fare with mean median of age and fare
print(f'Original age variable Variance {X_train["Age"].var()}')
print(f'Age variance after median Imputation {X_train["Age_median"].var()}')
print(f'Age variance after mean Imputation {X_train["Age_mean"].var()}')

print(f'Original Fare variable Variance {X_train["Fare"].var()}')
print(f'Fare variance after median Imputation {X_train["Fare_median"].var()}')
print(f'Fare variance after mean Imputation {X_train["Fare_mean"].var()}')


# plotting graph

fig = plt.figure()
ax = fig.add_subplot(111)
X_train['Age'].plot(kind = 'kde', ax = ax, color = 'red')
X_train['Age_median'].plot(kind = 'kde', ax = ax, color = 'green')
X_train['Age_mean'].plot(kind = 'kde', ax = ax, color = 'blue')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc = 'best')

fig = plt.figure()
ax1 = fig.add_subplot(111)
X_train['Fare'].plot(kind = 'kde', ax = ax1, color = 'red')
X_train['Fare_median'].plot(kind = 'kde', ax = ax1, color = 'green')
X_train['Fare_mean'].plot(kind = 'kde', ax = ax1, color = 'blue')
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc = 'best')

print(X_train.cov())
print(X_train.corr())


# check outliers - plot boxplot for this

X_train[['Age', 'Age_median', 'Age_mean']].plot()
X_train[['Fare', 'Fare_median', 'Fare_mean']].boxplot()



plt.show()