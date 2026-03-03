import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("student-mat.csv",sep=';')

# Create target
data["Result"] = data["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Features for website
features = ["studytime", "failures", "absences", "G2"]

X = data[features]
y = data["Result"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

# Save
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved.")
