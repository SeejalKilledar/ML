import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('world_trade_growth_clean.csv')

# Drop unnecessary identifier columns
df = df.drop(columns=['country_code', 'period', 'country_name'])

# Define X and y
X = df.drop('trade_growth_rate', axis=1)
y = df['trade_growth_rate']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Define column groups
categorical_cols = ['iso2_code', 'region']
ordinal_cols = ['income_group']
numeric_cols = ['year']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('ordinal', OrdinalEncoder(categories=[[
            'Not Classified','Low Income','Lower Middle Income',
            'Upper Middle Income','High Income'
        ]]), ordinal_cols),
        ('scale', MinMaxScaler(), numeric_cols)
    ],
    remainder='drop'   # 🔥 IMPORTANT
)

# Feature selection for regression
feature_selection = SelectKBest(score_func=f_regression, k=8)

# Model
model = DecisionTreeRegressor(random_state=0)

# Pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', feature_selection),
    ('model', model)
])

# Train
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
