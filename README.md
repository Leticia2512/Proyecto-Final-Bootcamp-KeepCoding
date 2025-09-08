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
├── 224x224      #Galeria Imágenes preprocesadas y formatos 224
├── 384x384      #Galeria Imágenes preprocesadas y formatos 384
├── app          #Aplicación final (API/Interfaz/RAG) 
├── archive      #Histórico de Scripts
├── arquitectura_entrenamientos #Creación de dataset final, Experimentos de aquitecturas y/o modelos y optimización de modelo neuronal final.
├── Data         #Fuentes de información(Csv,Dataset,Parqutes...)
├── EDA          #Análisis exploratorio del dataset(metadatos) e imagenes 
├── info_inicial #Fuentes de información (Recolección de Datasets para el proyecto)
├── MLFLow       #Inicialización MlFlow
├── mlruns       #Datos experimentos en MlFlow
├── ODIR-5K      #Imágenes dataset originales
├── requirements.txt #Librerias necesarias para el proyecto
├── meeting      #Reuniónes de equipo
└── README.md    # Este archivo
```
## Puesta en marcha.
Comandos a ejecutar desde consola:
    -activamos el API
```bash    
    fastapi dev app/main.py --reload
```    
    http://localhost:8501

    -activamos Streamlit
 ```bash     
    streamlit run app/app.py
```    
    http://127.0.0.1:8000/docs 

Una vez tengamos los servicios ejecutandose en el navegador interactuamos con la aplicación.

## Funciones para configuraciones extras.

-Transformacion de imagenes
```bash
   python preprocesado\imagenes\transform_img.py 224
```
    transformamos las imagenes a resolución 224x224

-Creación de dataset
```bash
    python arquitectura_entrenamientos\create_split_dataset.py
```
    creamos los dataset con las imagenes y los metadatos.

-Entrenamiento del modelo.
```bash
     python arquitectura_entrenamientos\red_neuronal_final.py
```    

-Tranformacion/limpieza pdf a txt.
```bash    
    python app\RAG\transform.py app\RAG\documentos\consenso_DMAE.pdf
```    
    se nos guarta el documento en app\RAG\Fixed

-Vectorizar.
```bash
    python app\RAG\chunks.py
``` 
    se nos generan dos archivos con extension JSon y Faiss(usado para el RAG)

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
