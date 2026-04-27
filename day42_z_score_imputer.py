"""
First way
1. Check the column is normally distributed/ almost normally distributed
2. Calculate mean+3 * std_deviation and mean - 3 * standard_deviation
3. either perform or trimming
4. If trimmin just remove all the outliers
5. If capping replace outlier value with min and max

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('placement.csv')
print(df.sample(5))

print("************  Step 1 ******************")
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])
plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])
plt.show()

print(f"Mean value of cgpa {df['cgpa'].mean()}")
print(f"Std value of cgpa {df['cgpa'].std()}")
print(f"Min value of cgpa {df['cgpa'].min()}")
print(f"Max value of cgpa {df['cgpa'].max()}")

print("************* Step 2 **************")
# finding boundary values
print(f"Highest allowed {df['cgpa'].mean()+ 3 * df['cgpa'].std()}")
print(f"Lowest allowed {df['cgpa'].mean()- 3 * df['cgpa'].std()}")

print("****** Fetch outliers **********")
print(df[(df['cgpa']>8.80) | (df['cgpa'] < 5.11)] )


# Trimming
new_df = df[(df['cgpa']<8.80) & (df['cgpa'] > 5.11)]
print(new_df.shape)

# second technique of trimming
"""
second way
1. Calculate Z-Score 
(Xi)^1 = (Xi - Mean)/ (Standard_Deviation)

2. Fetch the df z score value with 3 and - 3

3, And trim the df value > 3 and < -3

"""
df['cgpa_zscore'] = (df['cgpa'] - df['cgpa'].mean())/ df['cgpa'].std()
print(df.head())

print(df[df['cgpa_zscore'] > 3 ])
print(df[df['cgpa_zscore'] < -3 ])

print(df[(df['cgpa_zscore'] > 3) | (df['cgpa_zscore'] < -3) ])

# trimming

new_df = df[(df['cgpa_zscore'] < 3) | (df['cgpa_zscore'] > -3) ]
print(new_df)


# capping
print("****************** CAPPING ****************")
"""
1. Calculate upper limit
2. Calculate Lower limit
3. cgpa if > then upper limit, replace with upper limit
4. cgpa if < then lower limit, replace  with lower limit
5. If step 3 and 4 steps holds false, replace with cgpa as it is
"""

upper_limit = df['cgpa'].mean() + 3 * df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3 * df['cgpa'].std()

print(upper_limit, lower_limit)
print(df['cgpa'].min())
print(df['cgpa'].max())

df['cgpa'] = np.where(
    df['cgpa'] > upper_limit, upper_limit,
    np.where(df['cgpa'] < lower_limit, lower_limit,
    df['cgpa'])

)
print(df['cgpa'].min())
print(df['cgpa'].max())


