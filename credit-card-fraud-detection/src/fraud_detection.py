import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ===== PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

# Load dataset
data = pd.read_csv(DATA_PATH)

# -------- FEATURES --------
# Expected columns:
# Time, V1, V2, V3, V4, Amount, Class
X = data.drop("Class", axis=1)
y = data["Class"]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    classification_report(
        y_test, y_pred,
        zero_division=0
    )
)

# Save model
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("✅ Model saved at:", MODEL_PATH)
