import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------- Load Data -------------------- #
df = pd.read_csv("titanic.csv")

# Drop useless columns
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Split features & target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- Column Selection (SAFE) -------------------- #
numeric_cols = X_train.select_dtypes(include=np.number).columns
categorical_cols = X_train.select_dtypes(exclude=np.number).columns

# -------------------- Numeric Pipeline -------------------- #
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

# -------------------- Categorical Pipeline -------------------- #
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# -------------------- Combine -------------------- #
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# -------------------- Final Model -------------------- #
pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', DecisionTreeClassifier(
        max_depth=4,            # prevent overfitting
        min_samples_split=20,
        random_state=42
    ))
])

# -------------------- Train -------------------- #
pipe.fit(X_train, y_train)

# -------------------- Evaluate -------------------- #
train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# -------------------- Cross Validation -------------------- #
cv_score = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
print("Cross Validation Accuracy:", cv_score)
