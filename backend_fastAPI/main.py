from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # comunicación con Streamlit
# imports necesarios para el modelo
import numpy as np
from PIL import Image

from class_values import PatientData
from backend_functions import validate_image_jpg
import urllib.parse
import torch

app = FastAPI(
    title="Backend de modelo de Clasificación Ocular",         
    description="API para comunicarse con Streamlit y detección de enfermedades oculares",
    version="1.0.0"
)

# Muestra toda la información sobre la predicción que arroja nuestro modelo en base a los datos que nos dio.
@app.get("/prediction", summary="Predicción Ocular en base a los datos otorgados.")
async def get_ocular_prediction(patient: PatientData):

    age = patient.age
    gender = patient.gender
    image_url = patient.image_url

    result = 5 #Normal

    if age < 1 or age > 120:
        raise HTTPException(status_code=400, detail="Debes introducir una edad que corresponda entre 1 y 120 años")

    correct_url, status_url = validate_image_jpg(image_url) 

    if correct_url == False:
        raise HTTPException(status_code=404, detail=status_url)
        
    response = {"ocular_prediction": result}
   
    return response


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