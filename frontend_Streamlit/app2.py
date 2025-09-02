import streamlit as st
import requests
import json
from PIL import Image
import io
import csv
from datetime import datetime
import os

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000/prediction"

# Rutas a las carpetas y archivos
BASE_DIR = os.path.join(os.getcwd(), 'backend_fastAPI')
TEMP_IMAGES_DIR = os.path.join(BASE_DIR, "temp_images")
LOG_FILE = os.path.join(BASE_DIR, "prediction_log.csv")

# Función para guardar los datos de la predicción en un archivo CSV
def log_prediction(data):
    # Encabezados del CSV
    fieldnames = [
        "id", "fecha_hora", "edad", "genero", "ruta_imagen", 
        "clase_predicha", "probabilidades", "ruta_doc"
    ]
    
    # Verificar si el archivo ya existe
    file_exists = os.path.isfile(LOG_FILE)
    
    # Obtener el último ID para el auto-incremento
    next_id = 1
    if file_exists:
        with open(LOG_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Intenta encontrar la última fila del archivo
            try:
                for row in reader:
                    pass  # Este bucle leerá hasta la última fila
                if 'row' in locals():  # Verifica si la variable 'row' fue asignada
                    next_id = int(row['id']) + 1
            except StopIteration:
                # El archivo está vacío o solo tiene el encabezado
                pass

    # Preparar la fila para escribir en el CSV
    row_to_write = {
        "id": next_id,
        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "edad": data["age"],
        "genero": data["gender"],
        "ruta_imagen": data["image_url"],
        "clase_predicha": data["predicted_class"],
        "probabilidades": json.dumps(data["probabilities"]), # Convertir la lista a string JSON
        "ruta_doc": "---" 
    }
    
    # Escribir la fila en el archivo CSV
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Escribir el encabezado solo si el archivo es nuevo
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_to_write)

st.title("👁️ Ocular Prediction System")
st.write("Upload an eye image and provide patient information for prediction")

# Create form for user input
with st.form("prediction_form"):
    st.header("Patient Information")
    
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", ["male", "female"])
    
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Choose an eye image", type=["jpg", "jpeg"])
    
    submitted = st.form_submit_button("Make Prediction")

if submitted:
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Eye Image", use_container_width='content')
            
            # Crear el directorio para las imágenes temporales si no existe
            os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)

            # Guardar la imagen en la carpeta temp_images
            image_filename = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
            image_path = os.path.join(TEMP_IMAGES_DIR, image_filename)
            image.save(image_path)
            
            # Prepare the data for FastAPI
            data = {
                "age": age,
                "gender": gender,
                "image_url": image_path  # This should be the path your FastAPI can access
            }
            
            # Show loading spinner
            with st.spinner("Making prediction..."):
                # Send POST request to FastAPI
                response = requests.post(FASTAPI_URL, json=data)
            
            if response.status_code == 200:
                result = response.json()

                class_mapping = {
                    0: "Age-related Macular Degeneration",
                    1: "Cataract",
                    2: "Diabetic Retinopathy",
                    5: "Pathologic Myopia",
                    6: "Normal"
                }

                 # Preparar los datos para el log
                log_data = {
                    "age": age,
                    "gender": gender,
                    "image_url": image_path,
                    "predicted_class": class_mapping.get(result["predicted_class"], "UNKNOWN"),
                    "probabilities": result["probabilities"]
                }

                # Guardar el log en el archivo CSV
                log_prediction(log_data)
                
                st.success("✅ Prediction successful!")
                
                # Display results
                st.subheader("Prediction Condition")
                
                # Map class numbers to human-readable labels
                class_mapping = {
                    0: "Age-related Macular Degeneration",
                    1: "Catarat",
                    2: "Diabetic Retinopathy",
                    5: "Pathologic Myopia",
                    6: "Normal"
                }
                
                predicted_class = result["predicted_class"]
                probabilities = result["probabilities"]
                
                st.markdown(class_mapping.get(predicted_class, "UNKNOWN"))
                
                # Display probabilities
                st.subheader("Probability Distribution")
                
                for i, prob in enumerate(probabilities):
                    key = list(class_mapping.keys())[i]
                    class_name = class_mapping.get(key, f"Class {i}")
                    st.progress(prob, text=f"{class_name}: {prob:.2%}")
                
                # Show raw JSON data
                with st.expander("View Raw Response"):
                    st.json(result)
                    
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please upload an image first.")
