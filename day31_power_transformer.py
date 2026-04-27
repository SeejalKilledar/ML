import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.preprocessing import PowerTransformer

from day30_function_transform import X_train_transformed

"""
1. No null values no imputation required

"""



df = pd.read_csv('concrete.csv')
print(df.shape)
print(df.isnull().sum())

# check 0's and -ve values
print(df.describe()) # slag has 0 value

# train test split
X = df.drop(columns = ['strength'])
y = df['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2,
                                                    random_state= 42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
print(r2_score(y_test, y_predict))

#cross validation
lr = LinearRegression()
print(np.mean(cross_val_score(lr, X,y, scoring = 'r2')))

# for col in X_train.columns:
#     print(col)
#     plt.figure(figsize=(14,4))
#     plt.subplot(121)
#     sns.distplot(X_train[col], kde = True)
#     plt.title(col)
#
#     plt.figure(figsize=(14, 4))
#     plt.subplot(122)
#     stats.probplot(X_train[col], dist = 'norm', plot = plt)
#     plt.title(col)


    #plt.show()

# applying BOX- COX Transform
pt = PowerTransformer(method='box-cox')
X_trained_transform = pt.fit_transform(X_train+0.000001)
X_test_transformed = pt.transform(X_test+0.000001)
print(pd.DataFrame({'cols':X_train.columns, 'box-cox_lambdas':pt.lambdas_}))

lr = LinearRegression()
lr.fit(X_trained_transform, y_train)

y_predict2 = lr.predict(X_test_transformed)
print(r2_score(y_test, y_predict2))

pt = PowerTransformer(method='box-cox')
lr = LinearRegression()
y = df.iloc[:,-1]
print(f'YYYY {y}')
X_transform = pt.fit_transform(X_train+0.000001)
print(np.mean(cross_val_score(lr, X + 0.000001,y, scoring='r2')))

# before and after comparision for Box-Cox plot

X_train_transformed = pd.DataFrame(X_transform, columns=X_train.columns)

for col in X_train_transformed.columns:
    print(col)
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(X_train[col], kde = True)
    plt.title(col)


    plt.subplot(122)
    sns.distplot(X_train_transformed[col], kde = True)
    plt.title(col)

    #plt.show()

# apply Yeo-Johnson transform

pt1 = PowerTransformer()

X_trained_transform2 = pt1.fit_transform(X_train)
X_test_transform2 = pt1.fit_transform(X_test)

lr = LinearRegression()
lr.fit(X_trained_transform2, y_train)

y_predit3 = lr.predict(X_test_transform2)

print(r2_score(y_test, y_predit3))
