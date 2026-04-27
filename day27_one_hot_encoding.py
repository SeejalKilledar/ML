import numpy as np
import pandas as pd


df = pd.read_csv('Cars dataset.csv')
print(df.sample(5))

print(df['brand'].value_counts())
print(df['fuel'].value_counts())
print(df['owner'].value_counts())


# One Hot Encoding using Pandas
print("****** One Hot Encoding Using Pandas *******")
# get_dummies()
print(pd.get_dummies(df, columns=['fuel', 'owner']))

# 4 new Fuel columns + 5 new owner columns = In total 12 columns

# K-1 endcoding, removing 1st column
print(pd.get_dummies(df, columns=['fuel', 'owner'], drop_first=True))
# 3 new Fuel columns + 4 new owner columns = In total 10 columns

# While doing Machine Learning project, do not use Pandas, Use Scikit learn

# One Hot Encoding using Sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('selling_price', axis =1),
                                                    df['selling_price'],
                                                    train_size= 0.2,
                                                    random_state= 0
                                                    )
#print(X_train)

from sklearn.preprocessing import OneHotEncoder
#oe = OneHotEncoder(drop = 'first', sparse = False) #add to remove 1st column, Sparse = False (u dono have to convert toarray())
oe = OneHotEncoder(drop = 'first', dtype=np.int32)
# x_train_new and x_test_new has only Fuel and Owner,
x_train_new = oe.fit_transform(X_train[['fuel', 'owner']]).toarray() # this produces Saprse matrix
x_test_new = oe.fit_transform(X_test[['fuel', 'owner']]).toarray()
print(x_test_new.shape)
# x_train_new and x_test_new - should be appended wo brand and km_driven
print(np.hstack((X_train[['brand','km_driven']].values, x_train_new)).shape)


print("**************** OneHotEncoding with Top Categories *******************")
count = df['brand'].value_counts()
#print(count)

threshold = 100

repl = count[count<=threshold].index
#print(repl)

#print(pd.get_dummies(df['brand'].replace(repl, 'uncommon'), dtype=int))
