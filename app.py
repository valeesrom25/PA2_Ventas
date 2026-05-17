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

  # Predicciones
predicciones = modelo.predict(df)

# DataFrame
pred_df = pd.DataFrame({
    "Predicción de Ventas": predicciones
})

# Mostrar tabla
st.subheader("Predicciones")
st.dataframe(pred_df.head(20))

# Estadísticas
st.subheader("Resumen Estadístico")

st.write("Cantidad de predicciones:", len(predicciones))
st.write("Promedio:", round(predicciones.mean(), 2))
st.write("Máximo:", round(predicciones.max(), 2))
st.write("Mínimo:", round(predicciones.min(), 2))

# Gráfico de barras
st.subheader("Gráfico de Predicciones")
st.bar_chart(pred_df.head(20))

# Histograma
st.subheader("Distribución de Predicciones")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.hist(predicciones, bins=20)

ax.set_title("Distribución de Predicciones")
ax.set_xlabel("Ventas Predichas")
ax.set_ylabel("Frecuencia")

st.pyplot(fig)
