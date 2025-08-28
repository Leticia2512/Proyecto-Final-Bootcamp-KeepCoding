from fastapi import FastAPI, 
from fastapi.middleware.cors import CORSMiddleware  # comunicación con Streamlit
# imports necesarios para el modelo
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI(
    title="Backend de modelo de Clasificación Ocular",         
    description="API para comunicarse con Streamlit y detección de enfermedades oculares",
    version="1.0.0"
)

# Configuración de CORS 
# =============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # En producción, especifica dominios exactos
    allow_credentials=True,
    allow_methods=["*"],              # GET, POST, PUT, DELETE
    allow_headers=["*"],              # Todos los headers
)
# variables globales


# Funciones auxiliares

# Endpoint

#creo que para que funcione con streamlit debe ser:
#@app.post(datos_input)