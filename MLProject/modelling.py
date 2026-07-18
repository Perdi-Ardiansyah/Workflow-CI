import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def train_model():
    # Mengaktifkan autolog otomatis bawaan MLflow (Syarat Kriteria 2/3 Basic)
    mlflow.autolog()
    
    # Memuat Dataset Bersih
    df = pd.read_csv("namadataset_preprocessing.csv")
    
    # Transformasi Data Target
    le_target = LabelEncoder()
    y = le_target.fit_transform(df['difficulty'])
    
    # Transformasi Fitur
    X = df[['category', 'force_type', 'mechanic', 'body_part']].copy()
    for col in X.columns:
        X[col] = LabelEncoder().fit_transform(X[col])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Memulai Sesi Pelatihan
    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Workflow CI Berhasil! Akurasi Model: {accuracy}")

if __name__ == "__main__":
    train_model()