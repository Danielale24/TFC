import streamlit as st
import pandas as pd
import joblib
from surprise import SVD
from surprise import Dataset
from surprise import Reader

# Título y explicación en la interfaz
st.title("🛍️ Recomendador para E-commerce")
st.write("Selecciona un usuario para ver qué productos podría interesarle.")

# Subir archivos
uploaded_model = st.file_uploader("Sube el archivo del modelo SVD (.pkl)", type="pkl")
uploaded_ratings = st.file_uploader("Sube el archivo de calificaciones (.csv)", type="csv")

# Verifica si ambos archivos se han subido
if uploaded_model is not None and uploaded_ratings is not None:
    # Cargar el modelo SVD desde el archivo subido
    modelo = joblib.load(uploaded_model)

    # Cargar las calificaciones de los productos (ratings) desde el archivo subido
    ratings = pd.read_csv(uploaded_ratings, names=['user_id', 'product_id', 'rating', 'timestamp'])

    # Filtrar los usuarios disponibles
    usuarios = ratings['user_id'].unique()

    # Interfaz para seleccionar un usuario
    user_id = st.selectbox("Selecciona un usuario", usuarios)

    if st.button("Obtener recomendaciones"):
        recomendaciones = []

        # Generar recomendaciones para el usuario seleccionado
        for product_id in ratings['product_id'].unique():
            try:
                # Realizamos la predicción utilizando el modelo
                pred = modelo.predict(user_id, product_id)
                recomendaciones.append((product_id, pred.est))
            except Exception as e:
                # En caso de error, simplemente continuamos con el siguiente producto
                continue

        # Ordenar las recomendaciones por la puntuación predicha
        top_recomendaciones = sorted(recomendaciones, key=lambda x: x[1], reverse=True)[:5]

        st.subheader("🎯 Recomendaciones:")

        # Mostrar las recomendaciones en la interfaz de Streamlit
        for product_id, score in top_recomendaciones:
            st.markdown(f"**Producto ID: {product_id}** — Predicción: `{score:.2f}`")
else:
    st.warning("Por favor, sube el archivo del modelo y el archivo de calificaciones.")

