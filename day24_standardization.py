import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Social_Network_Ads.csv')
# print(df.head())

df = df.iloc[:,2:]
print(df.sample(5))


# Train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Purchased', axis = 1),
                                                    df['Purchased'], test_size= 0.3,
                                                    random_state= 0
                                                    )
print(X_train.shape) # (280, 2)
print(X_test.shape) # (120, 2)

# Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# the above X_train_scaled and X_test_scaled is in numpy array convert to Dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns= X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns= X_test.columns)

print(X_train_scaled)

print(scaler.mean_)

print(np.round(X_train.describe(),1))
print("***************************************")
print(np.round(X_train_scaled.describe(),1))

# Effects of Scaling - Plot the graph to check the difference
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize= (12,5))
ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
ax1.set_title("Before Scaling")
ax2.scatter(X_train_scaled['Age'], X_train_scaled['EstimatedSalary'], color = 'red')
ax2.set_title("After Scaling")

# age and estimated salary distribution
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize= (12,5))
ax1.set_title("Before Scaling")
sns.kdeplot(X_train['Age'], ax = ax1)
sns.kdeplot(X_train['EstimatedSalary'], ax = ax1)
ax2.set_title("After Scaling")
sns.kdeplot(X_train_scaled['Age'], ax = ax2)
sns.kdeplot(X_train_scaled['EstimatedSalary'], ax = ax2)

# age distribution
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize= (12,5))
ax1.set_title("Before Scaling")
sns.kdeplot(X_train['Age'], ax = ax1)
ax2.set_title("After Scaling")
sns.kdeplot(X_train_scaled['Age'], ax = ax2)

# salary distribution
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize= (12,5))
ax1.set_title("Before Scaling")
sns.kdeplot(X_train['EstimatedSalary'], ax = ax1)
ax2.set_title("After Scaling")
sns.kdeplot(X_train_scaled['EstimatedSalary'], ax = ax2)

#plt.show()


print("*****************************************************************")
print("********* Why scaling is important ******************")

# logistic regression requires scaling
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr_scaled = LogisticRegression()

lr.fit(X_train, y_train)
lr_scaled.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test)
y_pred_scaled = lr_scaled.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

print(f"Accuracy score {accuracy_score(y_test,y_pred)}")
print(f"Scaled Accuracy score {accuracy_score(y_test,y_pred_scaled)}")


# Decision tree does not require scaling
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt_scaled = DecisionTreeClassifier()

dt.fit(X_train, y_train)
dt_scaled.fit(X_train_scaled, y_train)

dt_predit = dt.predict(X_test)
dt_pred_scaled = dt_scaled.predict(X_test_scaled)

print(f"Accuracy score {accuracy_score(y_test,dt_predit)}")
print(f"Scaled Accuracy score {accuracy_score(y_test,dt_pred_scaled)}")
