import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv('train.csv', usecols=['GarageQual', 'FireplaceQu', 'SalePrice'])

print(df.head())

print(df.isnull().mean()*100)



print("***** GarageQual Caculations with Pandas ********** ")
df['GarageQual'].value_counts().plot(kind = 'bar')

print(df['GarageQual'].mode())

fig = plt.figure()
ax = fig.add_subplot(111)

df[df['GarageQual']=='TA']['SalePrice'].plot(kind = 'kde', ax = ax)
df[df['GarageQual'].isnull()]['SalePrice'].plot(kind = 'kde', ax = ax, color = 'red')

lines, labels = ax.get_legend_handles_labels()
labels = ['House with TA', 'Houses with NA']
ax.legend(lines, labels, loc = 'best')


# fill na with TA
temp = df[df['GarageQual']=='TA']['SalePrice']
df['GarageQual'].fillna('Missing', inplace = True)
df['GarageQual'].value_counts().plot(kind = 'bar')

fig = plt.figure()
ax1 = fig.add_subplot(111)
temp.plot(kind = 'kde', ax = ax1, color = 'blue')
df[df['GarageQual']=='TA']['SalePrice'].plot(kind = 'kde', ax = ax1)
lines, labels = ax1.get_legend_handles_labels()
labels = ['House with TA', 'Houses with NA']
ax1.legend(lines, labels, loc = 'best')


print("***** FireplaceQu Caculations with Pandas ********** ")
df['FireplaceQu'].value_counts().plot(kind = 'bar')
print(df['FireplaceQu'].mode())

fig = plt.figure()
ax = fig.add_subplot(111)
df[df['FireplaceQu']== 'Gd']['SalePrice'].plot(kind = 'kde', ax = ax)
df[df['FireplaceQu'].isnull()]['SalePrice'].plot(kind = 'kde', ax = ax, color = 'red')
lines, labels = ax.get_legend_handles_labels()
labels = ['House with GD', 'Houses with NA']
ax.legend(lines, labels, loc = 'best')



temp1 = df[df['FireplaceQu']== 'Gd']['SalePrice']
df['FireplaceQu'].fillna('Missing', inplace = True)
fig = plt.figure()
ax = fig.add_subplot(111)
temp1.plot(kind = 'kde', ax = ax, color = 'green')
df[df['FireplaceQu']== 'Gd']['SalePrice'].plot(kind = 'kde', ax = ax)
lines, labels = ax.get_legend_handles_labels()
labels = ['House with GD', 'Houses with NA']
ax.legend(lines, labels, loc = 'best')
plt.show()


print("************ Using Sklearn **********")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('SalePrice', axis = 1), df['SalePrice'],
                                                    test_size=0.2)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='constant', fill_value='Missing')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print(imputer.statistics_)