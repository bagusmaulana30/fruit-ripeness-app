import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import joblib
import os

from skimage.feature import graycomatrix, graycoprops

# ==========================================
# LOAD MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "knn_model.pkl"
    )
)

image_path = None

# ==========================================
# OPEN IMAGE
# ==========================================

def open_image():

    global image_path

    image_path = filedialog.askopenfilename(
        title="Pilih Gambar Pisang",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    if not image_path:
        return

    img = Image.open(image_path)
    img = img.resize((300, 300))

    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)
    image_label.image = photo

    result_box.delete("1.0", tk.END)
    result_box.insert(
        tk.END,
        "Klik tombol Analyze untuk memulai analisis."
    )


# ==========================================
# ANALYZE
# ==========================================

def analyze():

    global image_path

    if image_path is None:
        messagebox.showwarning(
            "Peringatan",
            "Silakan pilih gambar terlebih dahulu."
        )
        return

    img = cv2.imread(image_path)

    if img is None:
        messagebox.showerror(
            "Error",
            "Gagal membaca gambar."
        )
        return

    # Resize

    img = cv2.resize(
        img,
        (500, 500)
    )

    # Konversi HSV

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    # Threshold

    lower = (15, 30, 30)
    upper = (90, 255, 255)

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

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

    # Segmentasi

    result = cv2.bitwise_and(
        img,
        img,
        mask=mask
    )

    if np.count_nonzero(mask) == 0:
        messagebox.showerror(
            "Error",
            "Objek pisang tidak terdeteksi."
        )
        return

    # ======================================
    # FITUR WARNA
    # ======================================

    h, s, v = cv2.split(hsv)

    mean_h = np.mean(
        h[mask > 0]
    )

    mean_s = np.mean(
        s[mask > 0]
    )

    mean_v = np.mean(
        v[mask > 0]
    )

    # ======================================
    # FITUR TEKSTUR
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

    # ======================================
    # KLASIFIKASI
    # ======================================

    sample = pd.DataFrame(
        [[
            mean_h,
            mean_s,
            mean_v,
            contrast,
            energy,
            homogeneity
        ]],
        columns=[
            "Hue",
            "Saturation",
            "Value",
            "Contrast",
            "Energy",
            "Homogeneity"
        ]
    )

    prediction = model.predict(sample)[0]

    print("Prediksi :", prediction)

    hasil = f"""

=========================================
         HASIL ANALISIS CITRA
=========================================

Hue           : {mean_h:.2f}

Saturation    : {mean_s:.2f}

Value         : {mean_v:.2f}

Contrast      : {contrast:.2f}

Energy        : {energy:.4f}

Homogeneity   : {homogeneity:.4f}

=========================================

HASIL KLASIFIKASI

         {prediction.upper()}

=========================================
"""

    result_box.delete(
        "1.0",
        tk.END
    )

    result_box.insert(
        tk.END,
        hasil
    )


# ==========================================
# GUI
# ==========================================

root = tk.Tk()

root.title(
    "Deteksi Kematangan Pisang"
)

root.geometry("700x900")
root.configure(
    bg="#f2f2f2"
)

judul = tk.Label(
    root,
    text="DETEKSI KEMATANGAN PISANG",
    font=("Arial", 20, "bold"),
    bg="#f2f2f2"
)

judul.pack(
    pady=15
)

image_label = tk.Label(
    root,
    bg="#f2f2f2"
)

image_label.pack()

btn_open = tk.Button(
    root,
    text="Open Image",
    font=("Arial", 11),
    width=20,
    command=open_image
)

btn_open.pack(
    pady=8
)

btn_analyze = tk.Button(
    root,
    text="Analyze",
    font=("Arial", 11),
    width=20,
    command=analyze
)

btn_analyze.pack(
    pady=5
)

result_box = tk.Text(
    root,
    width=65,
    height=20,
    font=("Consolas", 11)
)

result_box.pack(
    pady=20
)

result_box.insert(
    tk.END,
    "Pilih gambar pisang kemudian klik Analyze."
)

root.mainloop()