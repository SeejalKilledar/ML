"""
This has both Numerical and Categorical data

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


print("******** Numerical ***********")
df = pd.read_csv('titanic.csv', usecols=['Age','Fare','Survived'])
print(df.head())
print(df.isnull().mean()*100)

X = df.drop(columns='Survived')
y = df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

X_train['Age_Imputed'] = X_train['Age']
X_test['Age_Imputed'] = X_test['Age']

print(X_train.head(10))

# print(X_train['Age_Imputed'][X_train['Age_Imputed'].isnull()].head(10))
# print(X_train['Age'].dropna().sample(X_train['Age'].isnull().sum()).values)

X_train['Age_Imputed'][X_train['Age_Imputed'].isnull()] = X_train['Age'].dropna().sample(X_train['Age'].isnull().sum()).values
X_test['Age_Imputed'][X_test['Age_Imputed'].isnull()] = X_test['Age'].dropna().sample(X_test['Age'].isnull().sum()).values

print(X_train.head(10))

sns.distplot(X_train['Age'], label = 'Original', hist= False)
sns.distplot(X_train['Age_Imputed'], label= 'Imputed', hist = False)
#
# plt.legend()


print(f'Original variables variance: {X_train["Age"].var()}')
print(f'Variance applied after Imputation: {X_train["Age_Imputed"].var()}')

print(X_train[['Age', 'Fare', 'Age_Imputed']].cov())
X_train[['Age', 'Age_Imputed']].boxplot() # for checking outliers

# code for when the user enters age, the fare should be same
#sampled_value = X_train['Age'].dropna().sample(1, random_state=int(observation['Fare']))


print("******** Categorical Data ********** ")
data = pd.read_csv('train.csv', usecols=['GarageQual','FireplaceQu', 'SalePrice'])
print(data.head(10))
print(data.isnull().mean()*100)

X = data
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)

X_train['GarageQual_Imputed']  = X_train['GarageQual']
X_test['GarageQual_Imputed']  = X_test['GarageQual']

X_train['FireplaceQu_Imputed']  = X_train['FireplaceQu']
X_test['FireplaceQu_Imputed']  = X_test['FireplaceQu']

print(X_train.head(10))

X_train['GarageQual_Imputed'][X_train['GarageQual_Imputed'].isnull()] = X_train['GarageQual'].dropna().sample(X_train['GarageQual'].isnull().sum()).values

X_test['GarageQual_Imputed'][X_test['GarageQual_Imputed'].isnull()] = X_test['GarageQual'].dropna().sample(X_test['GarageQual'].isnull().sum()).values


X_train['FireplaceQu_Imputed'][X_train['FireplaceQu_Imputed'].isnull()] = X_train['FireplaceQu'].dropna().sample(X_train['FireplaceQu'].isnull().sum()).values
X_test['FireplaceQu_Imputed'][X_test['FireplaceQu_Imputed'].isnull()] = X_test['FireplaceQu'].dropna().sample(X_test['FireplaceQu'].isnull().sum()).values
print(X_train.head(10))

#print(X_train['GarageQual'].dropna().sample(X_train['GarageQual'].isnull().sum()).values)

temp = pd.concat(
    [
        X_train['GarageQual'].value_counts() / len(X_train['GarageQual'].dropna()),
    X_train['GarageQual_Imputed'].value_counts() / len(X_train)

 ], axis = 1

)
temp.columns = ['Garage_original', 'Garage_imputed']
print(temp)
temp1 = pd.concat(
    [
        X_train['FireplaceQu'].value_counts() / len(X_train['FireplaceQu'].dropna()),
        X_train['FireplaceQu_Imputed'].value_counts() / len(df)

    ], axis = 1

)
temp1.columns = ['FireplaceQu_original', 'FireplaceQu_imputed']
print(temp1)


for category in X_train['FireplaceQu'].dropna().unique():
    sns.distplot(X_train[X_train['FireplaceQu'] == category]['SalePrice'],hist=False, label = category)

for category in X_train['FireplaceQu_imputed'].dropna().unique():
    sns.distplot(X_train[X_train['FireplaceQu_imputed'] == category]['SalePrice'],hist=False, label = category)
plt.show()