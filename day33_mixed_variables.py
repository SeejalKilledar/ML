import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    'titanic_mixed_variables.csv')

print(df.head())

print(df['number'].unique())

fig = df['number'].value_counts().plot.bar()
plt.show()


"""
Number col has numerical and categorical data,
1. create 2 cols, 
2. 1 numerical and 2nd categorical
3. when we receive Numerical data, fill numerical col with the num and categorical with NA
4. when we receive Categorical data, fill numerical col with the NA and categorical with the value

Below code will extract the numerical part and categorical part

"""

# extract Numerical part
df['number_numerical'] = pd.to_numeric(df['number'], errors='coerce',downcast='integer')
# extract Categorical part
df['number_categorical'] = np.where(df['number_numerical'].isnull(),df['number'], np.nan)


print(df.head())


"""
Cabin : 'C85' 'E31' 'C123'
"""

print(df['Cabin'].unique())

df['cabin_num'] = df['Cabin'].astype(str).str.extract(r'(\d+)')
df['cabin_cat'] = df['Cabin'].str[0]


df['ticket_num'] = df['Ticket'].apply(lambda s : s.split()[-1])
df['ticket_num'] = pd.to_numeric(df['ticket_num'], errors='coerce',downcast='integer')


df['ticket_cat'] = df['Ticket'].apply(lambda s : s.split()[0])
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(),np.nan,df['ticket_cat'])

print(df.head(100))



