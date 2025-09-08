# Sistema de Predicción Ocular  
En este repositorio encontraras una aplicación que combina la **inteligencia artificial** con el análisis de imágenes para ayudarte a comprender mejor el estado de tus ojos.  

Con solo una **imagen** y algunos **datos básicos del paciente**, el sistema es capaz de ofrecer:  
- Una predicción precisa sobre la condición ocular.  
- Información adicional de apoyo para comprender mejor cada diagnóstico.  

---

## Predicciones detalladas para la salud visual  

El modelo ha sido entrenado para identificar con gran precisión una variedad de **afecciones oculares comunes**. Actualmente, puede predecir:  

- **Degeneración Macular Relacionada con la Edad (DMAE):** Afecta la visión central, común en personas mayores.  
- **Catarata:** Opacificación del cristalino que causa visión borrosa.  
- **Retinopatía Diabética:** Daño a los vasos sanguíneos de la retina ocasionado por la diabetes.  
- **Miopía Patológica:** Forma severa de miopía que puede generar complicaciones graves.  
- **Ojo Normal:** Confirmación de que no se detectan alteraciones.  

---

## Más allá de la predicción  
 Nuestro sistema utiliza **RAG (Retrieval-Augmented Generation)** para buscar y recuperar información confiable sobre la condición predicha y ofrece al usuario detalles adicionales que enriquecen la comprensión del resultado.  

Con este enfoque, no solo **identificamos la afección**, sino que también **aportamos conocimiento y contexto** para una mejor toma de decisiones.  


## Estructura del repositorio
```
.
├── 224x224                       #Galeria Imágenes preprocesadas y formatos 224
├── 384x384                       #Galeria Imágenes preprocesadas y formatos 384
├── app                           #Aplicación final (API/Interfaz/RAG) 
├── archive                       #Histórico de Scripts
├── arquitectura_entrenamientos   #Creación de dataset final, Experimentos de aquitecturas y/o modelos y optimización de modelo neuronal final.
├── Data                          #Fuentes de información(Csv,Dataset,Parqutes...)
├── EDA                           #Análisis exploratorio del dataset(metadatos) e imagenes 
├── info_inicial                  #Fuentes de información (Recolección de Datasets para el proyecto)
├── MLFLow                        #Inicialización MlFlow
├── mlruns                        #Datos experimentos en MlFlow
├── ODIR-5K                       #Imágenes dataset originales
├── requirements.txt              #Librerias necesarias para el proyecto
├── meeting                       #Reuniónes de equipo
└── README.md                     #Este archivo
```
## Puesta en marcha

### 1) Activar la API (FastAPI)
Ejecuta el servidor de desarrollo:
```bash
fastapi dev app/main.py --reload
```
Documentación disponible en: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2) Activar la interfaz (Streamlit)
```bash
streamlit run app/app.py
```
Aplicación disponible en: [http://localhost:8501](http://localhost:8501)

> Una vez ambos servicios estén corriendo, puedes interactuar con la aplicación desde el navegador.

---

## 🛠️ Funciones para configuraciones extra

### Transformar imágenes a 224×224
```bash
python preprocesado\imagenes\transform_img.py 224
```

### Crear el dataset (imágenes + metadatos)
```bash
python arquitectura_entrenamientos\create_split_dataset.py
```

### Entrenar el modelo
```bash
python arquitectura_entrenamientos\red_neuronal_final.py
```

### Transformación / limpieza de PDF a TXT (RAG)
```bash
python app\RAG\transform.py app\RAG\documentos\consenso_DMAE.pdf
```
El TXT resultante se guarda en: `app\RAG\Fixed`

### Vectorizar (generar índices para RAG)
```bash
python app\RAG\chunks.py
```
Se generan dos archivos: uno **JSON** y otro **FAISS** (usado por el RAG).

## Autores

- **David**  
  📧 Contacto: *No disponible*

- **Javier**  
  📧 Contacto: javiluque78@gmail.com

- **Leticia**  
  📧 Contacto: leticia.c.morales@gmail.com

- **Miguel Ángel**  
  📧 Contacto: *No disponible*

- **Nauzet Fernández**  
  📧 Contacto: Nauzet.fdez@gmail.com

- **Sara**  
  📧 Contacto: sara.carcamo.r@gmail.com  

- **Sofía**  
  📧 Contacto: sofiagabian80@gmail.com


>  ¡Gracias por pasarte por aquí! 
