"""
1. Plot the graph check if the col is skewed
2. Check how much is the col skewed with .skew()
3. describe the col, to check 25% (25 percentile) and 75% (75 percentile) and other details
4. calculate 0.25 and 0.75 quantile using quantile funct
5. Calculate iqr (0.75 - 0.25)
6. Calculate upper limit : quantile 75 + 1.5 * iqr
             lower limit : quantile 25 - 1.5 * iqr
7. Calculate outliers for upper limit and lower limit
8. Apply trimming
9. Compare old plotting with new trimmed data
10. Capping
11. Compare old plotting with new capped data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('placement.csv')
print(df.sample(5))

print("***************** Step1 ************")
plt.figure(figsize=(16,5))
plt.subplot(121)
sns.distplot(df['cgpa'])
plt.subplot(122)
sns.distplot(df['placement_exam_marks'])

print("************** Step2 ************")
print(df['placement_exam_marks'].skew()) # far from 0
print(df['cgpa'].skew()) # near to 0 as it is normally distributed

print("***************** Step3 **********************")
print(df['placement_exam_marks'].describe())
sns.boxplot(df['placement_exam_marks'])

print("************ Step4 *********************")
percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)

print("************* Step5 **********************")
iqr = percentile75 - percentile25

print("*************** Step6 ***********************")
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

print("************** Step7 ************************")
print(df[df['placement_exam_marks'] > upper_limit])
print(df[df['placement_exam_marks'] < lower_limit])

print("************** Step8 *******************")
print(df.shape)
new_df = df[df['placement_exam_marks'] < upper_limit]
#new_df = df[(df['placement_exam_marks'] < upper_limit) & (df['placement_exam_marks'] > lower_limit)]
print(new_df.shape)

print("**************** Step9 ***********************")
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df['placement_exam_marks'])

print("*************** Step 10 **************************")
new_df_cap = df.copy()
new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit, upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit, lower_limit,
        new_df_cap['placement_exam_marks']
    )
)

print("**************** Step11 ************************")
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df_cap['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df_cap['placement_exam_marks'])

print(new_df_cap.shape)
plt.show()