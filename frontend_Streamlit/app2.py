import streamlit as st
import requests
import json
from PIL import Image
import io
import csv
from datetime import datetime
import os
import sys

import sys
import os

# Agregar la carpeta padre al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG")))

from rag_pipelinev2 import run_rag_fusion

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000/prediction"

# Rutas a las carpetas y archivos
BASE_DIR = os.path.join(os.getcwd(), 'backend_fastAPI')
TEMP_IMAGES_DIR = os.path.join(BASE_DIR, "temp_images")
TEMP_DOCS_DIR = os.path.join(BASE_DIR, "temp_docs")
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
        "ruta_doc": data.get("ruta_doc", "---")  
    }
    
    # Escribir la fila en el archivo CSV
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Escribir el encabezado solo si el archivo es nuevo
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_to_write)

def update_log_with_rag_doc(doc_filename):
    """Busca el último registro en el CSV y actualiza la columna 'ruta_doc'."""
    try:
        with open(LOG_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            st.warning("No se encontró ningún registro para actualizar.")
            return

        last_row = rows[-1]
        last_row["ruta_doc"] = doc_filename

        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "id", "fecha_hora", "edad", "genero", "ruta_imagen", 
                "clase_predicha", "probabilidades", "ruta_doc"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
    except FileNotFoundError:
        st.error(f"El archivo de log no existe: {LOG_FILE}")
    except Exception as e:
        st.error(f"Ocurrió un error al actualizar el log: {str(e)}")

def create_report_content(data):
    """Genera el contenido del reporte en formato Markdown."""
    report = f"""
# Reporte del Sistema de Predicción Ocular

---

## Información del Paciente
- **Edad:** {data.get('age', 'No disponible')}
- **Género:** {data.get('gender', 'No disponible')}

---

## Resultados de la Predicción
- **Condición Predicha:** {data.get('predicted_class', 'No disponible')}
- **Distribución de Probabilidades:**
"""
    probabilities = data.get('probabilities', {})
    for class_name, prob in probabilities.items():
        report += f"- **{class_name}:** {float(prob):.2%}\n"

    report += """
---

## Información Adicional (Obtenida de RAG)
"""
    if 'rag_result' in data and data['rag_result']:
        report += data['rag_result']
    else:
        report += "No se pudo obtener información adicional."

    return report

st.title("👁️ Ocular Prediction System")
st.write("Upload an eye image and provide patient information for prediction")

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'rag_result' not in st.session_state:
    st.session_state.rag_result = None

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

            os.makedirs(TEMP_DOCS_DIR, exist_ok=True)

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

                # Crear un nuevo diccionario con nombres de clases y probabilidades
                probabilities_list = result["probabilities"]
                prob_with_names = {
                    class_mapping.get(key, "UNKNOWN"): prob
                    for i, (key, value) in enumerate(class_mapping.items())
                    for prob in [probabilities_list[i]]
                }

                predicted_class_name = class_mapping.get(result["predicted_class"], "UNKNOWN")

                 # Preparar los datos para el log
                log_data = {
                    "age": age,
                    "gender": gender,
                    "image_url": image_filename,
                    "predicted_class": predicted_class_name,
                    "probabilities": json.dumps(prob_with_names)
                }

                # Guardar el log en el archivo CSV
                log_prediction(log_data)

                # Almacena todos los datos necesarios en el estado de sesión
                st.session_state.prediction_results = {
                    "image": image,
                    "age": age,
                    "gender": gender,
                    "predicted_class": predicted_class_name,
                    "probabilities": prob_with_names
                }

                class_mapping_rag = {
                    0: "dmae",
                    1: "catarata",
                    2: "retinopatia",
                    5: "miopia",
                    6: "Normal"
                }
                st.session_state.rag_prediction_name = class_mapping_rag.get(result["predicted_class"], "UNKNOWN")
                        
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                st.session_state.prediction_data = None
                
        except Exception as e:
            st.error(f"❌ Ocurrió un error: {str(e)}")
            st.session_state.prediction_data = None
    else:
        st.warning("⚠️ Por favor, sube una imagen primero.")

# --- Visualizar los resultados de la predicción y el botón de información adicional ---
if "prediction_results" in st.session_state and st.session_state.prediction_results:
    results = st.session_state.prediction_results
    
    st.success("✅ Predicción exitosa!")
    
    st.subheader("Condición de Predicción")
    st.markdown(results["predicted_class"])
    
    st.subheader("Distribución de Probabilidad")
    for class_name, prob in results["probabilities"].items():
        st.progress(prob, text=f"{class_name}: {float(prob):.2%}")
        
    st.markdown("---")
    st.subheader("Información Adicional")
    
    if st.button("Obtener más información sobre la condición"):
        with st.spinner("Buscando más información..."):
            rag_result = run_rag_fusion(
                st.session_state.rag_prediction_name,
                edad=results["age"],
                sexo=results["gender"]
            )

        # Guardar el resultado de RAG en un archivo y loguear la ruta
        if rag_result:
            doc_filename = f"reporte_RAG_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            doc_path = os.path.join(TEMP_DOCS_DIR, doc_filename)
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(rag_result)

            update_log_with_rag_doc(doc_filename)
        
        st.session_state.rag_result = rag_result
        
    if st.session_state.rag_result:
        st.success("Información obtenida correctamente.")
        st.markdown(st.session_state.rag_result)
    else:
        st.markdown("Presione el botón para obtener más información.")

    st.markdown("---")
    
    st.session_state.prediction_results['rag_result'] = st.session_state.rag_result
    report_content = create_report_content(st.session_state.prediction_results)
    
    st.download_button(
        label="📥 Exportar Reporte Completo",
        data=report_content,
        file_name=f"reporte_{results["predicted_class"]}_{results["gender"]}_{results["age"]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.md",
        mime="text/markdown"
    )