import pandas as pd

df = pd.read_csv('titanic.csv')

# asking questions
# How big is the data?
print(df.shape)

# how does the data look?
print(df.head())
print(df.sample(5))

# data type of the coloumns
print(df.info)

# any missing values
print(df.info)
print(df.isnull().sum())

# how does the data look mathematically
print(df.describe)

# Are there any duplicate values
print(df.duplicated().sum())

# How is the correlation with the columns
df.corr()
df.corr()['Survived']