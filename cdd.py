import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk
from tkinter import ttk, messagebox

# -----------------------------------------------------------
# STEP 1 — Load Kaggle Dataset
# -----------------------------------------------------------
def load_dataset():
    try:
        df = pd.read_csv("cardio_train.csv", sep=";")  # Kaggle dataset
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("ERROR: cardio_train.csv not found! Download from Kaggle and place in same folder.")
        exit()

    # Convert age from days → years
    df["age"] = (df["age"] / 365).astype(int)

    features = [
        "age", "gender", "height", "weight",
        "ap_hi", "ap_lo",
        "cholesterol", "gluc",
        "smoke", "alco", "active"
    ]

    X = df[features]
    y = df["cardio"]

    return X, y, features


# -----------------------------------------------------------
# STEP 2 — Train the ML model
# -----------------------------------------------------------
def train_model():
    X, y, features = load_dataset()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\nModel Accuracy: {acc:.4f}")

    joblib.dump(model, "cardio_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(features, "features.pkl")

    print("Model, scaler & features saved successfully!")

    return model, scaler, features


# Load or train model
try:
    model = joblib.load("cardio_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    print("Loaded saved model.")
except:
    model, scaler, features = train_model()


# -----------------------------------------------------------
# STEP 3 — Build Tkinter GUI
# -----------------------------------------------------------
def predict_disease():
    try:
        values = [
            float(age_entry.get()),
            int(gender_var.get()),
            float(height_entry.get()),
            float(weight_entry.get()),
            float(sys_entry.get()),
            float(dia_entry.get()),
            int(chol_var.get()),
            int(gluc_var.get()),
            int(smoke_var.get()),
            int(alco_var.get()),
            int(active_var.get())
        ]
    except:
        messagebox.showerror("Error", "Please fill all fields correctly!")
        return

    values_scaled = scaler.transform([values])
    prediction = model.predict(values_scaled)[0]

    if prediction == 1:
        result_label.config(text="⚠ High Risk of Cardiovascular Disease", fg="red")
    else:
        result_label.config(text="✔ No Cardiovascular Disease Detected", fg="green")


# GUI Window
root = tk.Tk()
root.title("Cardiovascular Disease Predictor (Kaggle)")
root.geometry("480x600")

title = tk.Label(root, text="Cardio Disease Prediction", font=("Arial", 18, "bold"))
title.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

# ------------------ INPUT FIELDS ------------------
label_font = ("Arial", 12)

tk.Label(frame, text="Age (years):", font=label_font).grid(row=0, column=0, sticky="w")
age_entry = tk.Entry(frame); age_entry.grid(row=0, column=1)

tk.Label(frame, text="Gender (1=Female, 2=Male):", font=label_font).grid(row=1, column=0, sticky="w")
gender_var = ttk.Combobox(frame, values=[1, 2], width=17); gender_var.grid(row=1, column=1)

tk.Label(frame, text="Height (cm):", font=label_font).grid(row=2, column=0, sticky="w")
height_entry = tk.Entry(frame); height_entry.grid(row=2, column=1)

tk.Label(frame, text="Weight (kg):", font=label_font).grid(row=3, column=0, sticky="w")
weight_entry = tk.Entry(frame); weight_entry.grid(row=3, column=1)

tk.Label(frame, text="Systolic BP:", font=label_font).grid(row=4, column=0, sticky="w")
sys_entry = tk.Entry(frame); sys_entry.grid(row=4, column=1)

tk.Label(frame, text="Diastolic BP:", font=label_font).grid(row=5, column=0, sticky="w")
dia_entry = tk.Entry(frame); dia_entry.grid(row=5, column=1)

tk.Label(frame, text="Cholesterol (1–3):", font=label_font).grid(row=6, column=0, sticky="w")
chol_var = ttk.Combobox(frame, values=[1, 2, 3], width=17); chol_var.grid(row=6, column=1)

tk.Label(frame, text="Glucose (1–3):", font=label_font).grid(row=7, column=0, sticky="w")
gluc_var = ttk.Combobox(frame, values=[1, 2, 3], width=17); gluc_var.grid(row=7, column=1)

tk.Label(frame, text="Smoking (0/1):", font=label_font).grid(row=8, column=0, sticky="w")
smoke_var = ttk.Combobox(frame, values=[0, 1], width=17); smoke_var.grid(row=8, column=1)

tk.Label(frame, text="Alcohol (0/1):", font=label_font).grid(row=9, column=0, sticky="w")
alco_var = ttk.Combobox(frame, values=[0, 1], width=17); alco_var.grid(row=9, column=1)

tk.Label(frame, text="Active (0/1):", font=label_font).grid(row=10, column=0, sticky="w")
active_var = ttk.Combobox(frame, values=[0, 1], width=17); active_var.grid(row=10, column=1)

# ------------------ PREDICT BUTTON ------------------
predict_btn = tk.Button(root, text="Predict", font=("Arial", 14), bg="blue", fg="white",
                        command=predict_disease)
predict_btn.pack(pady=20)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=10)

root.mainloop()
