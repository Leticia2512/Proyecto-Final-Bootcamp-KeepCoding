from fastapi import FastAPI, HTTPException
# imports necesarios para el modelo
import numpy as np
from PIL import Image

from class_values import PatientData, PredictionResult
from backend_functions import validate_image_jpg
from pathlib import Path
from predict_with_model import predictor

import os
import shutil

# Rutas a las carpetas
BASE_DIR = Path(__file__).resolve().parent
TEMP_IMAGES_DIR = BASE_DIR / "temp_images"

app = FastAPI(
    title="Backend modelo de Clasificación Ocular",         
    description="API para detección de enfermedades oculares",
    version="1.1.0"
)

def startup_event():
    """Crea la carpeta de imágenes temporales al iniciar la aplicación."""
    if not TEMP_IMAGES_DIR.exists():
        os.makedirs(TEMP_IMAGES_DIR)

@app.get("/")
def read_root(): 
    return {"message": "¡Bienvenido a la API de Clasificación Ocular!"}

# Muestra toda la información sobre la predicción que arroja nuestro modelo en base a los datos que nos dio.
@app.post("/prediction", response_model=PredictionResult, summary="Predicción Ocular en base a los datos otorgados.")
async def get_ocular_prediction(patient: PatientData):

    age = patient.age
    gender = patient.gender
    image_url = patient.image_url

    image_path = Path(image_url)

    meta_data = {"age": age, "gender": gender}

    result = 5 #Normal

    if age < 1 or age > 120:
        raise HTTPException(status_code=400, detail="Debes introducir una edad que corresponda entre 1 y 120 años")

    correct_url, status_url = validate_image_jpg(image_url) 

    if correct_url == False:
        raise HTTPException(status_code=404, detail=status_url)
        
    predicted_class, probabilities = predictor.predict(str(image_path), meta_data)

    KEEP_CLASSES = [0, 1, 2, 5, 6]

     # Formatear las probabilidades a una lista
    probabilities_list = probabilities

    response = {"predicted_class": predicted_class, "probabilities": probabilities_list}
   
    return response



