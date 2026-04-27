import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data_science_job.csv')
print(df.info())
print(df.head(5))

print(df.isnull().mean()*100)

"""
As per the data from (df.isnull().mean()*100), below columns have <5% hence 
cannot CCA cannot be applied
gender                    23.530640
major_discipline          14.683161
company_size              30.994885
company_type              32.049274

>5% , CCA can be applied
city_development_index     2.500261
enrolled_university        2.014824
education_level            2.401086
training_hours             3.998330
"""

print(df.shape)


# fetch cols with 0.05 (5% missing values)
"""
my_list = []
for var in df.columns:
    print(var)
    if (df[var].isnull().mean() < 0.05) and (df[var].isnull().mean() > 0):
        my_list.append(var)

print(my_list)

shortcut for the above code is value of col
"""
cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]
print(cols)
print(df[cols].sample(5))

"""
total data in df is 19158
and data in df[cols] after dropping is 17182

so if we divide this 17182/19158 = 0.89
hence 89% of data is saved for us to train

"""
print(len(df[cols].dropna()))
print(len(df))

data_remaining = len(df[cols].dropna())/len(df)

new_df = df[cols].dropna()
print(df.shape, new_df.shape)


# comparision by plotting at once
new_df.hist(bins=50, density = True, figsize=(12,12))


# plotting individually , training_hours, city_development_index and experience

# Training hours
fig = plt.figure()
ax = fig.add_subplot(111)
df['training_hours'].hist(bins = 50, ax = ax, density= True, color = 'red')
new_df['training_hours'].hist(bins = 50, ax = ax, density= True, color = 'green')


# city_development_index
fig = plt.figure()
ax1 = fig.add_subplot(111)
df['city_development_index'].hist(bins = 50, ax = ax1, density= True, color = 'red')
new_df['city_development_index'].hist(bins = 50, ax = ax1, density= True, color = 'green')
# dist plot
df['city_development_index'].plot.density(color = 'red')
new_df['city_development_index'].plot.density(color = 'green')

# experience
fig = plt.figure()
ax2 = fig.add_subplot(111)
df['experience'].hist(bins = 50, ax = ax2, density= True, color = 'red')
new_df['experience'].hist(bins = 50, ax = ax2, density= True, color = 'green')


"""

Categorical cols: enrolled_univerisity, education_level
For categorical data, the ratio after applying CCA should be same/approximately same
Step: 1. calculate the ratio of each category before CCA
example: for col education_level, calculate the ratio of category Graduate before applying CCA(same fpr other categories as well)  
Step 2:  calculate the ratio of each category after CCA

How to calculate ratio
value_count_of_col / length_of_dataframe 
df[enrolled_univerisity].value_counts()/len(df)

"""
temp = pd.concat(
    [
     df['enrolled_university'].value_counts()/len(df),
     new_df['enrolled_university'].value_counts()/len(df),

    ], axis = 1


)

temp.columns = ['enrolled_university_original', 'enrolled_university_cca']
print(temp)

temp1 = pd.concat(
    [
        df['education_level'].value_counts() / len(df),
        new_df['education_level'].value_counts() / len(df),

    ], axis=1

)
temp1.columns = ['education_level_original', 'education_level_cca']
print(temp1)







#plt.show()