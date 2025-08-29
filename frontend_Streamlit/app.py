# streamlit run app.py [-- script args] -> Comando para ejecutar la app

import streamlit as st
import requests -> creo que tenemos que usarlo para la comunicacion con el backend
import matplotlib.pyplot as plt # para la grafica de resultados
from matplotlib.figure import Figure

def configure_page() -> None: # Título y configuración de la pestaña de la página
    st.set_page_config(
        page_title="Detección de Enfermedades Oculares",
        page_icon="👁️",
        layout="wide",
        #initial_sidebar_state="auto"
    )

def configure_overview() -> None: # Descripción de la aplicación
    #st.markdown("<h1 style='text-align: center;'>👁️ Detección de Enfermedades Oculares</h1>", unsafe_allow_html=True)
    st.title("👁️ Detección de Enfermedades Oculares")
    st.write(
        """
        Esta aplicación utiliza un modelo de aprendizaje automático para detectar enfermedades oculares a partir de imágenes de escáneres oculares y datos de sexo y edad del paciente.
        """
    )

def configure_inputs() -> None: # Inputs de la aplicación
    sex = st.selectbox("Sexo del Paciente", options=["Masculino", "Femenino", "Otro"])
    age = st.slider("Edad del Paciente", min_value=1, max_value=120, value=30, step=1)
    image_file = st.file_uploader("Suba la Imagen del Escáner Ocular", type=["jpg", "jpeg", "png"])

    
response = requests.post("http://localhost:8000/predict/", files=image_files) # Envía la imagen al backend FastAPI

def create_plot(results_plot: plot_results_class) -> Figure: # Aqui ponemos una funcion que cree el plot (de matplotlib de probabilidades de las enfermedades de diagnostico.
    # funtion variables

    return fig


def main() -> None:
    configure_page()
    configure_overview()
    configure_inputs()

    st.sidebar.markdown("---")
    st.sidebar.header("Configuración del Modelo")
    model_url = st.sidebar.text_input("URL del Modelo FastAPI", "http://
    fig = create_plot(results_plot)
    st.pyplot(fig)

