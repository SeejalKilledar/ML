import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
flight = sns.load_dataset('flights')
titanic = pd.read_csv('titanic.csv')

print(tips.info())
print(iris.info())
print(flight.info())
print(titanic.info())


"""
1. Scatterplot : Numerical - Numerical
7. Pair Plot - Numerical - Numerical
8. Lineplot - Numerical - Numerical
2. Bar Plot    : Numerical - categorical
3. Box Plot - Numerical - Categorical
4. Distplot - Numerical - Categorical
5. HeatMap - Categorical - Categorical
6. Clustermap - categorical - categorical


"""

print("****** Scatterplot: Numerical - Numerical *******")
"""
1. Plot 2 numerical data on x axis and y axis - bivariate
2. Plot 3,4,5 numerical data on x axis and y axis - bivariate
"""
#sns.scatterplot(data = tips, x = 'total_bill', y = 'tip') # bivariate, plotted 2 columns
#sns.scatterplot(data = tips, x = 'total_bill', y = 'tip', hue= 'sex') # multivariate, plotted 3 columns
#sns.scatterplot(data = tips, x = 'total_bill', y = 'tip', hue= 'sex', style = 'smoker', size = 'size') # multivariate, plotted 5 columns



print("**** Bar Plot - Numerical - Categorical")

#sns.barplot(data = titanic, x = 'Pclass', y = 'Age') # bivariate, x axis will have categorical and y will have numerical
#sns.barplot(data bfvff"?= titanic, x = 'Pclass', y = 'Fare') # bivariate
# sns.barplot(data = titanic, x = 'Pclass', y = 'Fare', hue = 'Sex') #multivariate


print("**** Box Plot - Numerical - Categorical")
#sns.boxplot(data = titanic, x = 'Sex', y = 'Age') # bivariate
# sns.boxplot(data = titanic, x = 'Sex', y = 'Age', hue = 'Survived')

print('*** Distplot - Numerical - Categorical')
# sns.displot(titanic[titanic['Survived']==0]['Age'])
# sns.displot(titanic[titanic['Survived']==1]['Age'])


print("****** HeatMap - Categorical - Categorical *****")
print(pd.crosstab(titanic['Pclass'], titanic['Survived']))
# sns.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived']))
#print(titanic.groupby('Pclass').mean()['Survived']*100)

print("******** Clustermap - categorical - categorical")
print(pd.crosstab(titanic['SibSp'], titanic['Survived']))
# sns.clustermap(pd.crosstab(titanic['Parch'], titanic['Survived']))

print("*** Pair Plot - Numerical - Numerical ***")
# collection of scatter plots
# sns.pairplot(iris)
# sns.pairplot(iris, hue = 'species')

print("****** Lineplot - Numerical - Numerical")
# this is used when one of the columns is date, time or day, anything related to date
# covid, stocks

# new = flight.groupby('year').sum().reset_index()
# print(sns.lineplot(data = new, x = 'year', y = 'passengers'))

flight.pivot_table(values = 'passengers', index = 'month', columns = 'year')

sns.heatmap(flight.pivot_table(values = 'passengers', index = 'month', columns = 'year'))

plt.show()

