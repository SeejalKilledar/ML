# ==================================
# 1. Import Libraries
# ==================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==================================
# 2. Load Dataset
# ==================================

df = pd.read_csv("titanic.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==================================
# 3. Data Cleaning
# ==================================

# Fix Missing Values (NO inplace)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin (too many nulls)
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


# ==================================
# 4. Remove Outliers (Fare - IQR)
# ==================================

# Q1 = df['Fare'].quantile(0.25)
# Q3 = df['Fare'].quantile(0.75)
# IQR = Q3 - Q1

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]

# ==========================================================

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    cleaned_data = data[(data[column] >= lower) & (data[column] <= upper)]

    print(f"{column} Lower Bound:", lower)
    print(f"{column} Upper Bound:", upper)
    print(f"Rows removed:", len(data) - len(cleaned_data))

    return cleaned_data


df = remove_outliers_iqr(df, 'Fare')
df = remove_outliers_iqr(df, 'Age')

# ==================================
# 5. Scaling
# ==================================

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# ==================================
# 6. Correlation Heatmap (FIXED)
# ==================================

plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ==================================
# 7. Feature Selection
# ==================================

# Drop non-numeric unused columns
X = df.drop(columns=['Survived', 'Name', 'Ticket', 'PassengerId'], errors='ignore')
y = df['Survived']

selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(abs(X), y)

selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)

# ==================================
# 8. Visualisations
# ==================================

# Histogram
plt.hist(df['Age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot
plt.scatter(df['Age'], df['Fare'])
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# ==================================
# 9. Model Training
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==================================
# 10. Accuracy
# ==================================

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
