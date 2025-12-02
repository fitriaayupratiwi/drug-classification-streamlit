import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ======================
# Load Dataset Lokal
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("drug200.csv") 
    return df

df = load_data()

st.title("Sistem Pendukung Keputusan - KNN Drug Classification")

# ======================
# Preprocessing
# ======================

# Encode kolom kategorikal
le = LabelEncoder()
df_encoded = df.copy()
for col in ["Sex", "BP", "Cholesterol", "Drug"]:
    df_encoded[col] = le.fit_transform(df[col])

# Split dataset
X = df_encoded.drop("Drug", axis=1)
y = df_encoded["Drug"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ======================
# Model KNN
# ======================
st.sidebar.header("Pengaturan Model")
k = st.sidebar.slider("Nilai K", 1, 15, 5)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Prediksi dan Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Akurasi Model")
st.write(f"**Akurasi KNN (k={k}) = {acc:.2f}**")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# ======================
# Form Prediksi Baru
# ======================
st.subheader("Prediksi Obat untuk Pasien Baru")

age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", df["Sex"].unique())
bp = st.selectbox("Blood Pressure (BP)", df["BP"].unique())
chol = st.selectbox("Cholesterol", df["Cholesterol"].unique())
na_to_k = st.number_input("Na_to_K", min_value=0.0, max_value=50.0, value=10.0)

# Encode input baru
input_data = pd.DataFrame([{
    "Age": age,
    "Sex": le.fit(df["Sex"]).transform([sex])[0],
    "BP": le.fit(df["BP"]).transform([bp])[0],
    "Cholesterol": le.fit(df["Cholesterol"]).transform([chol])[0],
    "Na_to_K": na_to_k
}])

# Prediksi
pred = model.predict(input_data)[0]
drug_label = le.fit(df["Drug"]).inverse_transform([pred])[0]

st.markdown(f"### **Obat yang direkomendasikan: {drug_label}**")

# ======================
# Visualisasi
# ======================
st.subheader("Distribusi Drug Label")
fig, ax = plt.subplots()
df["Drug"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)
