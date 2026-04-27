import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic.csv')

"""
Categorical columns : Survived, Pclass, Sex, SibSp, Parch, Cabin, Embarked
Numerical columns   : Age, Fare

"""


print("*********** Categorical Plot (Count plot and Pie chart) *************")
# sns.countplot(x = 'Survived', data = df)
# sns.countplot(df['Survived'])
#print(df['Survived'].value_counts().plot(kind = 'bar'))
# sns.countplot(x = 'Survived', data = df)
# sns.countplot(x= 'Pclass', data = df)
# sns.countplot(x = 'Sex', data = df)
# sns.countplot(x = 'Cabin', data = df)
# sns.countplot(x = 'Embarked', data = df)

# df['Survived'].value_counts().plot(kind = 'pie', autopct = '%.2f')
# df['Cabin'].value_counts().plot(kind = 'pie', autopct = '%.2f')
# df['Embarked'].value_counts().plot(kind = 'pie', autopct = '%.2f')
# df['Sex'].value_counts().plot(kind = 'pie', autopct = '%.2f')


print("****** Numerical Data (Histogram/Distplot/BoxPlot) ********")
# plt.hist(df['Age'], bins = 50)
# sns.distplot(df['Age'])
sns.boxplot(x = df['Age'])
plt.show()

df['Age'].min()
df['Age'].max()
df['Age'].mean()