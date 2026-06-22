import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("churn.csv")

# Remove customerID
df = df.drop("customerID", axis=1)

# TotalCharges has 11 blank strings in this dataset -- force numeric first,
# this turns those blanks into proper NaN so they can be handled below
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values
df = df.ffill()

# Convert all categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object" or df[col].dtype == "str":
        df[col] = le.fit_transform(df[col].astype(str))

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")