from pydantic import BaseModel
from typing import List

# Clase sobre la información del paciente.
class PatientData(BaseModel):
    image_url: str
    age: int
    gender: str
 
# Modelo de datos para la respuesta
class PredictionResult(BaseModel):
    predicted_class: int
    probabilities: List[float]