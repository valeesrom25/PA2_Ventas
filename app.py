import streamlit as st
import pandas as pd
import joblib

# Título
st.title("Predicción de Ventas - Superstore")

# Datos del estudiante
st.write("Nombre: Valerie")
st.write("Código ISIL: 70959959")

# Link Colab
st.markdown(
    "[Abrir cuaderno en Google Colab](https://colab.research.google.com/drive/1DHtQ_B6gGc3xRJkCmW5B0MMt9VrFHmFj?usp=sharing)"
)

# Cargar modelo
modelo = joblib.load("modelos/modelo_random_forest.pkl")

# Subir archivo
archivo = st.file_uploader(
    "Sube tu archivo CSV",
    type=["csv"]
)

if archivo is not None:

    df = pd.read_csv(archivo, encoding='latin1')

    st.subheader("Vista previa")
    st.write(df.head())

    # Procesamiento
    columnas_eliminar = [
        "Sales",
        "Order ID",
        "Customer Name",
        "Product Name"
    ]

    for col in columnas_eliminar:
        if col in df.columns:
            df = df.drop(col, axis=1)

    df = pd.get_dummies(df)

    # Cargar columnas entrenamiento
    columnas_modelo = joblib.load("modelos/columnas.pkl")

    df = df.reindex(columns=columnas_modelo, fill_value=0)

    # Predicción
    predicciones = modelo.predict(df)

    st.subheader("Predicciones")
    st.write(predicciones)
