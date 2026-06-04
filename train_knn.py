import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Lokasi dataset

dataset_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset.csv"
)

# Load dataset

df = pd.read_csv(dataset_path)

# Feature dan Label

X = df[
    [
        "Hue",
        "Saturation",
        "Value",
        "Contrast",
        "Energy",
        "Homogeneity"
    ]
]

y = df["Label"]

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Training KNN

knn = KNeighborsClassifier(
    n_neighbors=7
)

knn.fit(
    X_train,
    y_train
)

# Prediksi

y_pred = knn.predict(X_test)

# Evaluasi

acc = accuracy_score(
    y_test,
    y_pred
)

print()
print("===============================")
print("HASIL TRAINING")
print("===============================")

print("Akurasi :", round(acc * 100, 2), "%")

print()
print(classification_report(
    y_test,
    y_pred
))

# Simpan model

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "knn_model.pkl"
)

joblib.dump(
    knn,
    model_path
)

print()
print("Model berhasil disimpan.")
print(model_path)