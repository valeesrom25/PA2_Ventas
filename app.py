import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# TÍTULO
# -----------------------------
st.title("Predicción de Ventas - Superstore")

# -----------------------------
# DATOS DEL ESTUDIANTE
# -----------------------------
st.write("Nombre: Valerie")
st.write("Código ISIL: 70959959")

# -----------------------------
# LINK COLAB
# -----------------------------
st.markdown(
    "[Abrir cuaderno en Google Colab](https://colab.research.google.com/drive/1DHtQ_B6gGc3xRJkCmW5B0MMt9VrFHmFj?usp=sharing)"
)

# -----------------------------
# CARGAR MODELO
# -----------------------------
modelo = joblib.load("modelos/modelo_random_forest.pkl")

# -----------------------------
# SUBIR CSV
# -----------------------------
archivo = st.file_uploader(
    "Sube tu archivo CSV",
    type=["csv"]
)

# -----------------------------
# SI SE SUBE ARCHIVO
# -----------------------------
if archivo is not None:

    try:

        # Leer CSV
        df = pd.read_csv(archivo, encoding='latin1')

        # Vista previa
        st.subheader("Vista previa del Dataset")
        st.dataframe(df.head())

        # Tamaño dataset
        st.write("Dimensiones del dataset:", df.shape)

        # -----------------------------
        # PROCESAMIENTO
        # -----------------------------
        columnas_eliminar = [
            "Sales",
            "Order ID",
            "Customer Name",
            "Product Name"
        ]

        for col in columnas_eliminar:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Convertir categóricas
        df = pd.get_dummies(df)

        # -----------------------------
        # CARGAR COLUMNAS DEL MODELO
        # -----------------------------
        columnas_modelo = joblib.load("modelos/columnas.pkl")

        # Ajustar columnas
        df = df.reindex(columns=columnas_modelo, fill_value=0)

        # -----------------------------
        # PREDICCIONES
        # -----------------------------
        predicciones = modelo.predict(df)

        # DataFrame predicciones
        pred_df = pd.DataFrame({
            "Predicción de Ventas": predicciones
        })

        # -----------------------------
        # TABLA
        # -----------------------------
        st.subheader("Predicciones")
        st.dataframe(pred_df.head(20))

        # -----------------------------
        # RESUMEN ESTADÍSTICO
        # -----------------------------
        st.subheader("Resumen Estadístico")

        st.write("Cantidad de predicciones:", len(predicciones))
        st.write("Promedio:", round(predicciones.mean(), 2))
        st.write("Máximo:", round(predicciones.max(), 2))
        st.write("Mínimo:", round(predicciones.min(), 2))

        # -----------------------------
        # GRÁFICO DE BARRAS
        # -----------------------------
        st.subheader("Gráfico de Predicciones")

        st.bar_chart(pred_df.head(20))

        # -----------------------------
        # HISTOGRAMA
        # -----------------------------
        st.subheader("Distribución de Predicciones")

        fig, ax = plt.subplots()

        ax.hist(predicciones, bins=20)

        ax.set_title("Distribución de Predicciones")
        ax.set_xlabel("Ventas Predichas")
        ax.set_ylabel("Frecuencia")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
