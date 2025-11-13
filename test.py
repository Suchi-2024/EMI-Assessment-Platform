import joblib

scaler = joblib.load("models/scaler.joblib")

print("Scaler trained on", len(scaler.feature_names_in_), "features:")
print(list(scaler.feature_names_in_))