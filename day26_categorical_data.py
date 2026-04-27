# encoding categorical data | label encoding and ordinal encodng

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("https://github.com/YBIFoundation/Dataset/raw/main/Customer%20Purchase.csv")
#print(df.head())


"""
Columns : Customer ID,  Age,  Gender, Education,   Review, Purchased
Categorical Col : Gender, Review, Education, Purchased
Label Encoder/ Nominal Categorical : Purchased 
One Hot encoder/Nominal Categorical : Gender
Ordinal Encoder/Ordinal categorical : Review, Education
"""

# In this we will only encode Review, Education and Purchased
# Gender will be encoded in Column Transformer section


# extract Review Education and Purchased

df = df.iloc[:,3:]
print(df.head( ))

# Ordinal - Review and Education
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                df.drop('Purchased', axis = 1),
                                df['Purchased'], test_size= 0.2,
                                random_state= 0

)

#print(X_test.shape)
print(X_train)


from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['School','UG','PG'],['Poor','Average','Good']])
oe.fit(X_train)
X_train = oe.transform(X_train)
X_test = oe.transform(X_test)
print(X_train)
print(oe.categories)


# Purchased - Label Encoding
from sklearn.preprocessing import LabelEncoder
le = Encoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

print(y_train)