import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('heart.csv')
df.head()
X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

rf = RandomForestClassifier(oob_score=True)
rf.fit(X_train, y_train)
print(rf.oob_score_)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))