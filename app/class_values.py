from pydantic import BaseModel

# Clase sobre la información del paciente.
class PatientData(BaseModel):
    image_url: str
    age: int
    gender: str
 
