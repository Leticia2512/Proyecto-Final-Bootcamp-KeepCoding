
# Resumen de Experimentos — Modelo Multiclase CNN + Metadatos (ODIR-5K)

## Descripción general
- **Arquitectura:** ResNet18 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout).
- **Optimización:** AdamW (`lr` según experimento), `weight_decay` (regularización), `label_smoothing=0.05`.
- **Desbalanceo:** `class_weights` en la pérdida (sin sampler).
- **Tracking (MLflow):**
  - Métricas: `train_loss`, `train_acc`, `train_f1m`, `val_loss`, `val_acc`, `val_f1m`, y métricas finales de test.
  - Artefactos: mejor checkpoint, `classification_report` y `confusion_matrix` (val/test).

---

## Comparativa de configuración

### Hiperparámetros y cambios principales

| Parámetro           | Exp.1 (Local, 23/08/25) | Exp.2 (Colab, 24/08/25) | Exp.3 (Solo imágenes, 25/08/25) | Exp.4 (Metadatos de nuevo, 25/08/25) | Comentario |
|---------------------|--------------------------|--------------------------|----------------------------------|---------------------------------------|------------|
| **Batch size**      | 32                       | 64                       | 32                               | 32                                    | 64 en Colab para estabilidad y velocidad. |
| **Épocas**          | 15                       | 20                       | 25                               | 20                                    | Se alargó para observar convergencia. |
| **LR**              | 1e-4                     | 1e-5                     | 1e-4                             | 1e-4                                  | En Exp.2 se bajó para controlar sobreajuste. |
| **Weight decay**    | 2e-4                     | 5e-4                     | 5e-4                             | 5e-4                                  | Más regularización desde Exp.2. |
| **Label smoothing** | 0.05                     | 0.05                     | 0.05                             | 0.05                                  | Constante. |
| **Metadatos**       | Sí                       | Sí                       | No                               | Sí                                     | Exp.3 prueba sólo imágenes. |

### Data Augmentation

| Transformación    | Exp.1      | Exp.2      | Exp.3      | Exp.4                                   |
|-------------------|------------|------------|------------|-----------------------------------------|
| **Flip**          | p=0.2      | p=0.5      | p=0.5      | p=0.5                                   |
| **Rotación**      | ±15°       | ±10°       | ±10°       | — (Affine:rotación, desplazamiento y escalado)                          |
| **RandomAffine**  | —          | —          | —          | (±7°, transl. 2%, escala ±5%)        |
| **ColorJitter**   | —          | —          | —          | (±10%, `hue` 0.02)                   |
| **RandomErasing** | —          | p=0.1      | p=0.1      | p=0.25                  |
| **Normalize**     | ImageNet   | ImageNet   | ImageNet   | ImageNet                                |

> **Nota:** Exp.4 introduce un pipeline más completo y **clínicamente conservador**.

---

## Resultados globales (macro-F1 / accuracy)

| Conjunto    | Exp.1   | Exp.2   | Exp.3   | Exp.4   |
|-------------|---------|---------|---------|---------|
| **Train**   | 0.876 / 0.851 | 0.907 / 0.899 | 0.773 / 0.735 | **0.943 / 0.922** |
| **Val**     | 0.529 / 0.574 | 0.546 / 0.582 | 0.492 / 0.503 | **0.718 / 0.716** |
| **Test**    | 0.562 / 0.608 | 0.573 / 0.595 | 0.528 / 0.546 | **0.750 / 0.724** |


---

- Rendimiento por clase (Exp.4, test):

	**Clase 1 (n=30):** F1=0.889 (prec=0.848, rec=0.933)  
	**Clase 5 (n=20):** F1=0.889 (prec=0.800, rec=1.000)  
	**Clase 6 (n=287):** F1=0.768  
	**Clase 2 (n=161):** F1=0.610 (principal cuello de botella)  
	**Clase 0 (n=27):** F1=0.596 (alta varianza)  

---

## Cambios clave entre entrenamientos

- **Exp.1 → Exp.2:**  
  - ↓ `lr` (1e-5)  
  - ↑ `batch_size` (64)  
  - ↑ regularización (`wd` 5e-4)  
  - ↑ augmentación (flip/rot/erasing)  
  **→ Mejora ligera en F1 val/test (+0.017 / +0.011).**

- **Exp.2 → Exp.3 (solo imágenes + data augmentation):**  
  - Se eliminan metadatos.  
  **→ Bajón de F1 y accuracy en val/test. Confirma que los metadatos aportan.**

- **Exp.3 → Exp.4 (metadatos + data augmentación):**  
  - Clases usadas en Exp.4: keep_classes = [0, 1, 2, 5, 6]
  - Se reintroducen metadatos.  
  - Se añaden `RandomAffine` + `ColorJitter` suaves.  
  **→ Mejora notable: macro-F1 test 0.75, recall macro 0.78.**


### Notas específicas del Experimento 4 (25/08/25)

- Reducir ruido/clases con poco soporte y concentrar el modelado en las categorías con mayor representación clínica en el dataset actual.
- **Implicación**: las métricas (val/test) de Exp.4 **no son directamente comparables** con Exp.1–3 a nivel de macro-F1 global si estos incluían más clases; aun así, la comparación **intra-clases coincidentes** (0,1,2,5,6) sí es válida y muestra una **mejora clara** frente a configuraciones previas.

---

## Conclusiones breves
- **Overfitting evidente:** gap grande train vs val/test.  
- **Los metadatos aportan valor:** quitarlos baja claramente el rendimiento (Exp.3).  
- **Clase 2 sigue siendo el cuello de botella:** podría requierir técnicas específicas (Focal Loss, oversampling, mixup).  
- **Exp.4 es el que presenta mejores resultados:** macro-F1=0.75 en test con recall macro alto (0.78).  



















