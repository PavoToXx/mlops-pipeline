import joblib
import os

s = joblib.load('models/scaler.pkl')
m = joblib.load('models/model.pkl')

print(f'scaler size: {os.path.getsize("models/scaler.pkl")} bytes')
print(f'model size: {os.path.getsize("models/model.pkl")} bytes')
print(f'scaler n_features: {s.n_features_in_}')
print(f'model n_features: {m.n_features_in_}')
print(f'scaler feature_names: {list(s.feature_names_in_)}')
