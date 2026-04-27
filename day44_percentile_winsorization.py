import numpy as np
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('weight-height.csv')
print(df.head())

print(df.shape)
print(df['Height'].describe())

sns.distplot(df['Height'])
sns.boxplot(df['Height'])

upper_limit = df['Height'].quantile(0.99)
lower_limit = df['Height'].quantile(0.01)

print(df[(df['Height'] >= upper_limit) | (df['Height'] <= lower_limit)])

new_df = df[(df['Height'] <= upper_limit) & (df['Height'] >= lower_limit)]
print(new_df)
sns.displot(new_df['Height'])
sns.boxplot(new_df['Height'])


# capping
df['Height']=np.where(df['Height'] >= upper_limit, upper_limit,
         np.where(df['Height']<= lower_limit, lower_limit, df['Height']))
sns.displot(new_df['Height'])
sns.boxplot(new_df['Height'])

plt.show()