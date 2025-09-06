# Resumen Experimentos_Leticia (ODIR-5K · Modelo Multiclase CNN + Metadatos)

> **Arquitectura común**: ResNet18 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5).  
> **Optimización**: AdamW (con `weight_decay`), `label_smoothing=0.05`.  
> **Desbalanceo**: `class_weights` en la pérdida (sin sampler).  
> **Tracking (MLflow)**: `train_loss`, `train_acc`, `train_f1m`, `val_loss`, `val_acc`, `val_f1m`, y métricas finales en test; artefactos de `classification_report` y `confusion_matrix` (val/test).

---

## Exp_1 — *Experimento_Leticia_ResNet18*
- **Fecha**: 23/08/2025  
- **Run ID**: `eb3b723076f3496d895e74a4af59b742`  
- **Ejecución**: 1.4 h (CPU)  
- **Configuración**:  
  - **Transform (train)**: `RandomHorizontalFlip(p=0.2)`, `RandomRotation(15)`, normalización ImageNet.  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=2e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=6.
- **Resultados**:  
  - **Validación**: Acc **0.579**, F1 **0.510**  
  - **Test**: Acc **0.608**, F1 **0.562**
- **Evaluación**: **overfitting** (train F1 0.876 vs val F1 0.510). Clases minoritarias (p.ej., clase 4) con desempeño bajo. Se requiere más regularización o data augmentation.

---

## Exp_2 — *Experimento_Leticia_ResNet18*
- **Fecha**: 24/08/2025  
- **Run ID**: `df6bf684983c436680940f450870c1c7`  
- **Ejecución**: 54.4 min (GPU)  
- **Cambios respecto a Exp_1**:  
  - `batch_size` **32→64** (tiene mejor estabilidad de gradiente en GPU).  
  - `epochs` **15→20**, para convergencia
  - **LR** **1e-4→1e-5** (controlar sobreajuste).  
  - `weight_decay` **2e-4→5e-4** (más regularización).  
  - **Data Augmentation**: `Flip(p=0.5)` + `Rot(10)` + `RandomErasing(0.1)`.  
  - `num_workers=2` (entorno Colab).  
  - Se añade la métrica `prec_m` al logging.
- **Resultados**:  
  - **Validación**: Acc **0.582**, F1 **0.546**  
  - **Test**: Acc **0.595**, F1 **0.573**, Prec **0.551**, Rec **0.609**
- **Evaluación**: Ligera mejora frente a Exp_1, con algo menos de **overfitting** y mejor **recall** global gracias a la regularización y técnicas de augmentación.

---

## Exp_3 — *Experimento_Leticia_ResNet18 (solo con imágenes)*
- **Fecha**: 25/08/2025  
- **Run ID**: `06a0c25ec51c47948bc3ebe0cf0d8ae3`  
- **Ejecución**: 53.44 min (GPU)  
- **Cambios respecto a Exp_2**:  
  - **Sin metadatos** (flujo solo con imágenes).  
  - `epochs=25`; augmentation: `Flip(p=0.5)` + `Rot(10)` + `RandomErasing(0.1)`.  
  - **Hiperparámetros**: `batch_size=32`, `lr=1e-4`, `weight_decay=5e-4`, `num_workers=2`, `label_smoothing=0.05`, `num_classes=8`.
- **Resultados**:  
  - **Validación**: Acc **0.483**, F1 **0.467**  
  - **Test**: Acc **0.546**, F1 **0.528**, Prec **0.484**, Rec **0.610**
- **Evaluación**: Quitar metadatos **empeora** el rendimiento (menor F1 val/test).

---

## Exp_4 — *Experimento_4_Leticia_ResNet18_Top5*
- **Fecha**: 25/08/2025  
- **Run ID**: `3189a256d9b544f5906f41cd35cfc46e`  
- **Ejecución**: 1.6 h (CPU)  
- **Cambios respecto a Exp_3**:  
  - Se **reintroducen metadatos**.  
  - **Reducción de clases** a **[0,1,2,5,6]** → `num_classes=5`.  
  - `epochs=20`, **augmentation** enriquecido: `RandomAffine(±7°, ±2% shift, 0.95–1.05 scale)` + `ColorJitter(0.10/0.10/0.10/0.02)` + `Flip(0.5)` + `RandomErasing(0.25)`.
- **Resultados**:  
  - **Validación**: Acc **0.710**, F1 **0.726**  
  - **Test**: Acc **0.724**, F1 **0.750**, Prec **0.722**, Rec **0.784**
- **Evaluación**: **Salto sustancial** de rendimiento al centrar el problema en clases más frecuentes y mejorar la augmentación. Más **robustez** y mejor equilibrio entre clases. Aunque se sigue existiendo overfitting.

---

## Exp_5 — *Experimento_4_Leticia_ResNet18_Top5* (mejoras de entrenamiento)
- **Fecha**: 26/08/2025  
- **Run ID**: `70a48d006d5d4368b50d1d92a28be053`  
- **Ejecución**: 1.9 h (CPU)  
- **Cambios respecto a Exp_4**:  
  - **Scheduler** `ReduceLROnPlateau` (menor LR cuando no mejora).  
  - **AMP** (Mixed Precision) habilitado cuando hay GPU.    
  - **Logging extra**: LR por época, frecuencias de clase del train, **matriz de confusión normalizada**, predicciones (`.npz`).
- **Resultados**:  
  - **Validación**: Acc **0.702**, F1 **0.738**  
  - **Test**: Acc **0.720**, F1 **0.775**, Prec **0.772**, Rec **0.780**
- **Evaluación**:  Hay una ligera mejora sobre Exp_4. El scheduler estabiliza la convergencia; mejor **F1 y recall**. Se evalúa como el mejor run.

---

## Tabla resumen de métricas

| Exp | Fecha      | #Clases | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 | Test Prec | Test Rec |
| --: | ---------- | :-----: | :-------: | :------: | :-----: | :----: | :------: | :-----: | :-------: | :------: |
|   1 | 23/08/2025 |    8    |   0.851   |   0.876  |  0.579  |  0.510 |   0.608  |  0.562  |  0.570\*  |   0.571. |
|   2 | 24/08/2025 |    8    |   0.899   |   0.907  |  0.582  |  0.546 |   0.595  |  0.573  |   0.551   |   0.609  |
|   3 | 25/08/2025 |    8    |   0.735   |   0.773  |  0.483  |  0.467 |   0.546  |  0.528  |   0.484   |   0.610  |
|   4 | 25/08/2025 |    5    |   0.022   |   0.943  |  0.710  |  0.726 |   0.724  |  0.750  |   0.722   |   0.784  |
|   5 | 26/08/2025 |    5    |   0.943   |   0.960  |  0.702  |  0.738 |   0.720  |  0.775  |   0.772   |   0.780  |


---

## Hiperarámetros principales 

| Exp | Batch | Epochs | LR inicial | Weight Decay | Augmentación principal                     | Dropout | Scheduler | Optimizer |
| --: | :---: | :----: | :--------: | :----------: | ------------------------------------------ | :-----: | :-------: | :-------: |
|   1 |   32  |   15   |    1e-4    |     2e-4     | Flip(0.2)+Rot(15)                          |   0.5   |     No    |   AdamW   |
|   2 |   64  |   20   |    1e-5    |     5e-4     | Flip(0.5)+Rot(10)+Erasing(0.1)             |   0.5   |     No    |   AdamW   |
|   3 |   32  |   25   |    1e-4    |     5e-4     | Flip(0.5)+Rot(10)+Erasing(0.1)             |   0.5   |     No    |   AdamW   |
|   4 |   32  |   20   |    1e-4    |     5e-4     | Affine+ColorJitter+Flip(0.5)+Erasing(0.25) |   0.5   |     No    |   AdamW   |
|   5 |   32  |   20   |    1e-4    |     5e-4     | Affine+ColorJitter+Flip(0.5)+Erasing(0.25) |   0.5   |   **Sí**  |   AdamW   |


---

## Conclusiones

- Los **Metadatos** sí tienen valor en el modelo, retirarlos (Exp_3) redujo el F1; su inclusión mejora la discriminación entre clases próximas.  
- La **Regularización y augmentación**, como rotaciones moderadas, erasing y jitter redujeron el sobreajuste observado en Exp_1, aunque no lo suficiente.
- Limitar el modelo al entrenamiento con las clases **[0,1,2,5,6]** (Exp_4–5) incrementó notablemente **F1** y estabilidad inter-clase.  
- La **Optimización dinámica**: `ReduceLROnPlateau` (Exp_5) aportó mejoras adicionales en **F1** y **recall**.  
- El **Mejor resultdao** se confirma que se ha obtenido en el **Exp_5**, pero sin alcanzar el objetivo de eliminar el **overfitting** persistente.




