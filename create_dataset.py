import cv2
import numpy as np
import pandas as pd
import os

from skimage.feature import graycomatrix, graycoprops

# ==========================================
# Lokasi folder training
# ==========================================

base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "training"
)

labels = ["mentah", "setengah", "matang"]

data = []

print("========================================")
print("MEMBUAT DATASET")
print("========================================")
print("Folder Training :")
print(base_folder)
print()

# ==========================================
# Proses setiap folder
# ==========================================

for label in labels:

    folder = os.path.join(base_folder, label)

    print("Membaca folder :", folder)

    if not os.path.exists(folder):
        print("Folder tidak ditemukan!")
        continue

    for file in os.listdir(folder):

        if not (
            file.lower().endswith(".jpg")
            or file.lower().endswith(".jpeg")
            or file.lower().endswith(".png")
        ):
            continue

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            print("Gagal membaca :", file)
            continue

        # Resize
        img = cv2.resize(img, (500, 500))

        # RGB -> HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Thresholding
        lower = (15, 30, 30)
        upper = (90, 255, 255)

        mask = cv2.inRange(hsv, lower, upper)

        # Morphology
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Segmentasi objek
        result = cv2.bitwise_and(
            img,
            img,
            mask=mask
        )

        # ======================================
        # Ekstraksi Fitur Warna
        # ======================================

        h, s, v = cv2.split(hsv)

        if np.count_nonzero(mask) == 0:
            continue

        mean_h = np.mean(h[mask > 0])
        mean_s = np.mean(s[mask > 0])
        mean_v = np.mean(v[mask > 0])

        # ======================================
        # Ekstraksi Fitur Tekstur (GLCM)
        # ======================================

        gray = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2GRAY
        )

        glcm = graycomatrix(
            gray,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        contrast = graycoprops(
            glcm,
            "contrast"
        )[0, 0]

        energy = graycoprops(
            glcm,
            "energy"
        )[0, 0]

        homogeneity = graycoprops(
            glcm,
            "homogeneity"
        )[0, 0]

        # Simpan ke list

        data.append([
            mean_h,
            mean_s,
            mean_v,
            contrast,
            energy,
            homogeneity,
            label
        ])

# ==========================================
# Simpan ke CSV
# ==========================================

df = pd.DataFrame(
    data,
    columns=[
        "Hue",
        "Saturation",
        "Value",
        "Contrast",
        "Energy",
        "Homogeneity",
        "Label"
    ]
)

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset.csv"
)

df.to_csv(
    output_path,
    index=False
)

print()
print("========================================")
print("DATASET BERHASIL DIBUAT")
print("========================================")
print("Lokasi :", output_path)
print("Jumlah Data :", len(df))
print()

print(df.head())