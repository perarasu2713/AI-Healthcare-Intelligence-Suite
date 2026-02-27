import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =====================================================
# LOAD DATASET
# =====================================================
data = pd.read_csv("dataset/symptoms/dataset.csv")

print("Dataset Loaded Successfully")
print("Original Shape:", data.shape)

# =====================================================
# REMOVE MISSING VALUES (VERY IMPORTANT FIX)
# =====================================================
data = data.dropna()

print("After Removing NaN Rows:", data.shape)

# =====================================================
# SEPARATE FEATURES AND TARGET
# =====================================================
X = data.iloc[:, :-1].copy()   # use copy to avoid warnings
y = data.iloc[:, -1].copy()

# =====================================================
# ENCODE FEATURE COLUMNS (IF ANY ARE OBJECT TYPE)
# =====================================================
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# =====================================================
# ENCODE TARGET LABELS (DISEASE NAMES)
# =====================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Number of Disease Classes:", len(label_encoder.classes_))

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# MODEL ACCURACY
# =====================================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# =====================================================
# SAVE MODEL + ENCODER
# =====================================================
joblib.dump(model, "models/disease_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Model and Encoder saved successfully!")
