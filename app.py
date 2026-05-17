import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------------
# TÍTULO
# -----------------------------------
st.title("Predicción de Ventas - Superstore")

# -----------------------------------
# DATOS DEL ESTUDIANTE
# -----------------------------------
st.write("Nombre: Valerie")
st.write("Código ISIL: 70959959")

# -----------------------------------
# LINK COLAB
# -----------------------------------
st.markdown(
    "[Abrir cuaderno en Google Colab](https://colab.research.google.com/drive/1DHtQ_B6gGc3xRJkCmW5B0MMt9VrFHmFj?usp=sharing)"
)

# -----------------------------------
# CARGAR MODELO
# -----------------------------------
modelo = joblib.load("modelos/modelo_random_forest.pkl")

# -----------------------------------
# SUBIR ARCHIVO CSV
# -----------------------------------
archivo = st.file_uploader(
    "Sube tu archivo CSV",
    type=["csv"]
)

# -----------------------------------
# SI EL USUARIO SUBE ARCHIVO
# -----------------------------------
if archivo is not None:

    try:

        # -----------------------------------
        # LEER CSV
        # -----------------------------------
        df = pd.read_csv(archivo, encoding='latin1')

        # Guardar copia original
        df_original = df.copy()

        # -----------------------------------
        # VISTA PREVIA
        # -----------------------------------
        st.subheader("Vista previa del Dataset")
        st.dataframe(df.head())

        # Tamaño dataset
        st.write("Dimensiones del dataset:", df.shape)

        # -----------------------------------
        # PROCESAMIENTO
        # -----------------------------------
        columnas_eliminar = [
            "Sales",
            "Order ID",
            "Customer Name",
            "Product Name"
        ]

        for col in columnas_eliminar:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Convertir variables categóricas
        df = pd.get_dummies(df)

        # -----------------------------------
        # CARGAR COLUMNAS DEL MODELO
        # -----------------------------------
        columnas_modelo = joblib.load("modelos/columnas.pkl")

        # Ajustar columnas
        df = df.reindex(columns=columnas_modelo, fill_value=0)

        # -----------------------------------
        # REALIZAR PREDICCIONES
        # -----------------------------------
        predicciones = modelo.predict(df)

        # -----------------------------------
        # CREAR RESULTADO FINAL
        # -----------------------------------
        resultado = df_original.copy()

        resultado["Ventas Predichas"] = predicciones

        # -----------------------------------
        # EXPLICACIÓN
        # -----------------------------------
        st.subheader("Ventas estimadas por el modelo")

        st.write("""
        El modelo Random Forest estima el valor de ventas para cada pedido del dataset cargado.
        Cada predicción representa una estimación de ventas basada en las características del pedido,
        como categoría, región, descuentos y cantidad de productos.
        """)

        # -----------------------------------
        # MOSTRAR TABLA
        # -----------------------------------
        columnas_mostrar = [
            "Category",
            "Region",
            "Quantity",
            "Discount",
            "Ventas Predichas"
        ]

        columnas_existentes = [
            col for col in columnas_mostrar
            if col in resultado.columns
        ]

        st.dataframe(
            resultado[columnas_existentes].head(20)
        )

        # -----------------------------------
        # MÉTRICAS VISUALES
        # -----------------------------------
        st.subheader("Resumen Estadístico")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Promedio",
            round(predicciones.mean(), 2)
        )

        col2.metric(
            "Máxima Venta",
            round(predicciones.max(), 2)
        )

        col3.metric(
            "Mínima Venta",
            round(predicciones.min(), 2)
        )

        # -----------------------------------
        # HISTOGRAMA
        # -----------------------------------
        st.subheader("Distribución de Ventas Predichas")

        st.write("""
        El siguiente gráfico muestra cómo se distribuyen las ventas predichas por el modelo.
        Se puede observar en qué rangos se concentran la mayoría de ventas estimadas.
        """)

        fig, ax = plt.subplots()

        ax.hist(predicciones, bins=20)

        ax.set_title("Distribución de Ventas Predichas")
        ax.set_xlabel("Ventas Predichas")
        ax.set_ylabel("Frecuencia")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
