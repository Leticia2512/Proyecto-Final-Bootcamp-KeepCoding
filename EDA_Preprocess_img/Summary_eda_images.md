# Análisis Exploratorio de Imágenes (EDA) del dataset ODIR-5K

---

## 1. Objetivo

El objetivo de este análisis exploratorio es evaluar la **estructura** y la **calidad visual** de las imágenes de fondo de ojo (retinografías) de ambos ojos por paciente, presentes en el dataset **`full_df.csv`** de ODIR-5K, con el fin de identificar patrones, inconsistencias y posibles problemas que puedan influir en el rendimiento de un modelo basado en redes convolucionales.  

La revisión abarca:

- Análisis de dimensiones y formatos.  
- Verificación de integridad.  
- Inspección de la orientación de las imágenes.  
- Cálculo de métricas de calidad visual.  
- Evaluación por clase diagnóstica.  

Todo ello orientado a **definir el preprocesamiento más adecuado**.

---

## 2. Metodología

### 2.1 Carga y exploración inicial de las imágenes

Se importa el dataset en formato **CSV** y se exploran sus dimensiones, así como la estructura, tipos de datos y valores nulos:

- 6.392 registros distribuidos en 19 columnas.  
- Estructura coherente y bien organizada, integrando tanto información clínica como referencias cruzadas a las imágenes reales por paciente (`Left-Fundus`, `Right-Fundus`, `filepath`, `filename`).  
- Todas las columnas están completas, no presentan valores nulos.  

---

### 2.2 Análisis del tamaño y orientación de las imágenes

A partir de la función `get_image_sizes()`:

- Se recorren todas las imágenes y se extraen sus dimensiones (alto, ancho, canales).  
- Se identifican los tamaños más frecuentes y se verifica que no hay imágenes corruptas.  

**Resultados:**

- Gran variabilidad en las dimensiones, más de 15 tamaños diferentes.  
- Tamaño más común: **1728×2592×3** con 3.964 imágenes.  
- Otros tamaños frecuentes: **1536×2048×3 (978 imágenes)** y **1728×2304×3 (784 imágenes)**.  

**Implicaciones:**

- Necesario aplicar un **redimensionado uniforme** durante el preprocesamiento:  
  - Entrada consistente para el modelo (ResNet, EfficientNet requieren resolución fija).  
  - Reducción de coste computacional, reduciendo el tamaño de las imágenes más grandes.  
  - Evitar errores en los DataLoaders al trabajar con batches. 

No se identificaron imágenes corruptas → todos los archivos están disponibles para su procesamiento posterior.  

**Inspección visual aleatoria (`show_random_images()`):**

- Se observó variabilidad en orientación y leves rotaciones.  
- Recomendación: aplicar **data augmentation** para mejorar la generalización del modelo, o garantizar que todas las imágenes se encuentren alineadas en la misma dirección anatómica.  

**Distribución de tamaños y clases:**

- Gráfico de barras con los 15 tamaños más frecuentes → evidencia heterogeneidad.  
- Distribución por clase diagnóstica:  
  - Clases **D y N dominan (25–33%)**.  
  - Otras clases en proporciones mucho menores (3–6%). 

Se observa que las clases están bastante desbalanceadas. 

---

### 2.3 Análisis de color de las imágenes

El análisis busca evaluar variaciones en **brillo, contraste y distribución del color**, con el objetivo de considerar su impacto en las etapas posteriores de preprocesamiento, especialmente en lo relativo a la normalización y la aplicación de técnicas de data augmentation.

- En patologías como **retinopatía diabética** y **edema macular diabético**, los canales **rojo** y **verde** son especialmente relevantes (microhemorragias, exudados).  

**Funciones implementadas:**

- `compute_grayscale_histogram()`  
  - Calcula histograma promedio normalizado de intensidades en escala de grises.  
  - Intensidades medias están entre **50–150**, correspondientes a la región central de la retina, donde se concentra la mayor 	parte de la información clínica relevante.  
  - Intensidades bajas → áreas periféricas oscuras o fondo negro (sin valor diagnóstico).  

Análisis de los canales de color Rojo, Verde y Azul (RGB) para examinar la distribución de intensidades y posibles desequilibrios cromáticos en las imágenes:

- `compute_rgb_histogram()`  
  - Analiza distribución de intensidades RGB.  
  - Normalización a porcentaje.  
  - Resultado: picos cercanos a 0 → fuerte presencia de fondo negro.  

**Hallazgos:**

- El fondo negro podría inducir al modelo a aprender patrones irrelevantes.  
- Canal azul: menor variabilidad e intensidad relativa → aporta menos información diagnóstica.  
- Recomendación: aplicar `ColorJitter()` para resaltar estructuras anatómicas relevantes. Esta técnica permite ajustar de forma    aleatoria parámetros como el brillo, el contraste y la saturación, lo cual puede facilitar la detección de las diferentes patologías 

---

### 2.4 Análisis de calidad de las imágenes

Objetivo: detectar **bajo contraste, sobreexposición o iluminación desigual**.  

**Umbrales definidos:** `dark`, `bright`, `low_contrast`, `high_contrast`.  

**Funciones implementadas:**

- `analyze_image()` → calcula brillo y contraste por imagen, marcando flags de calidad.  
- `analyze_dataset()` → calcula estadísticas globales para cada categoría diagnóstica.  
- `plot_stats()` → histogramas de brillo y contraste indicando los umbrales.

**Resultados:**

- Imágenes oscuras: **7,4%**.  
- Bajo contraste: **2,9%**.  
- Brillantes: **2,0%**.  
- Alto contraste: **1,3%**.  

Conclusión: la mayoría de las imégenes tienen una calidad aceptables. Se podría valorar usar CLAHE en el preprocesamiento, pero de  forma selectiva, no global.

**Validación visual:**  
`display_examples()` muestra ejemplos por categoría. Para cada imagen se calcula y anota el brillo y el contraste, lo que permite una inspección cualitativa rápida de los casos marcados por los umbrales, confirmando los resultados de forma cualitativa.  

---

### 2.5 Distribución de imágenes por clase diagnóstica según calidad

Objetivo: identificar si determinadas **patologías están más afectadas por baja calidad**, y así justificar la aplicación de técnicas de preprocesamiento específicas para las clases diagnósticas más afectadas. 

**Procedimiento:**

- Creación de un dataframe con `filename`, métricas de brillo/contraste y flags de calidad (is_dark, is_bright, is_low_contrast,  is_high_contrast) según los umbrales previamente establecidos.  
- Expansión de la columna `labels` en indicadores binarios, devolviendo quality_df con ‘ID’, ‘filename’, métricas y etiquetas para análisis por patología.  
- Cálculo de % de imágenes problemáticas por clase diagnóstica, crea un DataFrame ordenado por porcentaje y se dibuja un gráfico de barras para clase, añadiendo una línea de media global como referencia. 

**Resultados:**

- **Glaucoma:** mayor proporción de imágenes oscuras y bajo contraste.  
- **Cataratas (C)** y **Fondo normal (N):** ligeramente más oscuras que la media.  
- **Cataratas (C)** y **Macular disease (M):** valores más altos de imágenes brillantes y alto contraste.  

Se podria valorar usar técnicas de **normalización de brillo/contraste** por clase en el preprocesaiento en lugar de eliminar imágenes.  

**Validaciones adicionales:**

- `plot_by_disease_and_category()` → visualizar ejemplos por clase y calidad.  
- Boxplot → brillo medio por clase diagnóstica, diferenciando casos positivos (con enfermedad) y negativos (sin enfermedad).  
- Test de Welch (no asume varianzas iguales): se compara el brillo medio entre imágenes con y sin cada patología (N–O):
	- glaucoma significativamente más oscuro; cataratas/miopía más brillantes.  

---

## 3. Conclusiones

### Imágenes oscuras (~7,4%)

- No eliminar en bloque (riesgo de sesgo por clase, ej. glaucoma).  
- Aplicar ajustes selectivos (`ColorJitter(brightness)`) o ponderar su contribución con sample_weight/WeightedRandomSampler en lugar de descartarlas.

### Imágenes con bajo/alto contraste

- Bajo contraste: 2,9%.  
- Alto contraste: 1,3%.  
- Valorar usar **CLAHE selectivo** en las marcadas como low_contrast y high_contrast ajustando clip limit para evitar artefactos. Nunca aplicar globalmente.  

### Tamaño de entrada

- Variabilidad alta → usar tamaños moderados para evitar coste computacional:  
  - 224×224 (ResNet18/MobileNet).  
  - 380×380 (EfficientNet-B4).  

### Estrategia de redimensionado

- `Resize + CenterCrop` (mantiene proporciones y región central).  
- O bien resize con padding para cubrir FOV completo.  

### Simetría y Data Augmentation

- Asegurar orientación izquierda/derecha correcta.  
- Usar `RandomHorizontalFlip` y rotaciones leves (`RandomRotation`) para mayor robustez.  

### Aumento cromático

- Dado el % bajo de incidencias (2,9% bajo contraste; 2,0% brillantes), aplicar `ColorJitter` con intensidad moderada y, preferiblemente, de forma condicionada a las imágenes marcadas como problemáticas.

### Balanceo de clases

- Usar `WeightedRandomSampler` o pérdidas ponderadas (`pos_weight`).  
- Split estratificado por paciente para preservar proporciones.  

### Normalización RGB

- Normalización global por canal (media/std del train).
- Alternativa: valores de **ImageNet** si se usan modelos preentrenados.  
