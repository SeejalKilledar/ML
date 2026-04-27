import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



df = pd.read_csv('wine_data.csv', usecols=[0,1,2])
print(df)

# sns.kdeplot(df['alcohol'])
# sns.kdeplot(df['malic_acid'])


fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize= (12,5))
sns.kdeplot(df['alcohol'], ax = ax1)
sns.kdeplot(df['malic_acid'], ax = ax2)
sns.scatterplot(data = df, x = 'alcohol', y = 'malic_acid',  hue = 'class_label', ax = ax3)


# Normaliztion - MinMax starts from here
# split train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('class_label', axis = 1),
                                                    df['class_label'],
                                                    test_size= 0.3,
                                                    random_state= 0
                                                    )

print(X_train.shape)
print(X_test.shape)

# Min Max
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled= scaler.transform(X_test)
# above data is in numpy array, hence converting it to Dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns= X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns= X_test.columns)

print(np.round(X_train.describe(),1))
print("**********************************************")
print(np.round(X_train_scaled.describe(),1))

fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize= (12,5))
ax1.set_title("Before scaling")
ax1.scatter(data = X_train, x = 'alcohol', y = 'malic_acid',  c = y_train)
ax2.set_title("After scaling")
ax2.scatter(data = X_train_scaled, x = 'alcohol', y = 'malic_acid',  c = y_train)


# kde plot
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize= (12,5))
ax1.set_title("Before scaling")
sns.kdeplot(X_train['alcohol'], ax = ax1)
sns.kdeplot(X_train['malic_acid'], ax = ax1)
ax2.set_title("After scaling")
sns.kdeplot(X_train_scaled['alcohol'], ax = ax2)
sns.kdeplot(X_train_scaled['malic_acid'], ax = ax2)

# individual - alcohol
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize= (12,5))
ax1.set_title("Before scaling")
sns.kdeplot(X_train['alcohol'], ax = ax1)
ax2.set_title("After scaling")
sns.kdeplot(X_train_scaled['alcohol'], ax = ax2)

# individual - malic acide.
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize= (12,5))
ax1.set_title("Before scaling")
sns.kdeplot(X_train['malic_acid'], ax = ax1)
ax2.set_title("After scaling")
sns.kdeplot(X_train_scaled['malic_acid'], ax = ax2)


plt.show()