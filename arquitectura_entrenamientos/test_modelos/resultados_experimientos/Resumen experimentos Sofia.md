# Resumen Experimentos_Sofia (ODIR-5K · Modelo Multiclase CNN + Metadatos)
 
> **Optimización**: AdamW (con `weight_decay`), `label_smoothing=0.05`.  
> **Desbalanceo**: `class_weights` en la pérdida (sin sampler).  
> **Tracking (MLflow)**: `train_loss`, `train_acc`, `train_f1m`, `val_loss`, `val_acc`, `val_f1m`, y métricas finales en test; artefactos de `classification_report` y `confusion_matrix` (val/test).

---

## Exp_1 — *resnet50_mm_20250825_115103*
- **Modelo**: ResNet50 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5). 
- **Fecha**: 25/08/2025 11:51 
- **Run ID**: `e4b95cede875432c9ca98500eff74af7`  
- **Ejecución**: 3.9 h (CPU)  
- **Configuración**:  
  - **Transform (train)**: `RandomHorizontalFlip(p=0.2)`, `RandomRotation(15)`, normalización ImageNet.  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=2e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=6.
- **Resultados**:  
  - **Entrenamiento**: Acc **0.864**, Loss **0.735**, F1 **0.882** 
  - **Validación**: Acc **0.610**, Loss **1.786**, F1 **0.518**  
  - **Test**: Acc **0.644**, Loss **1.166**, F1 **0.587**
- **Evaluación**:  

---

## Exp_2 — *efficientnetb3_mm_20250824_142437*
- **Modelo**: EfficientNet_B3 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5). 
- **Fecha**: 24/08/2025 14:24 
- **Run ID**: `90323cdd3c5b4dfd8e4644b0d7b1917a`  
- **Ejecución**: 3.3 h (CPU)  
- **Configuración**:  
  - **Transform (train)**: `RandomHorizontalFlip(p=0.2)`, `RandomRotation(15)`, normalización ImageNet.  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=2e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=6.
- **Resultados**:  
  - **Entrenamiento**: Acc **0.808**, Loss **0.787**, F1 **0.835** 
  - **Validación**: Acc **0.547**, Loss **1.695**, F1 **0.525**  
  - **Test**: Acc **0.608**, Loss **1.075**, F1 **0.573**
- **Evaluación**:  

---

## Exp_3 — *efficientnetb0_mm_20250824_181816*
- **Modelo**: EfficientNet_B0 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5). 
- **Fecha**: 24/08/2025 18:18
- **Run ID**: `0540f0daa7894ee0b18c8aa901d733dc`  
- **Ejecución**: 1.4 h (CPU)  
- **Configuración**:  
  - **Transform (train)**: `RandomHorizontalFlip(p=0.2)`, `RandomRotation(15)`, normalización ImageNet.  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=2e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=6.
- **Resultados**:  
  - **Entrenamiento**: Acc **0.781**, Loss **0.834**, F1 **0.816** 
  - **Validación**: Acc **0.599**, Loss **1.607**, F1 **0.552**  
  - **Test**: Acc **0.619**, Loss **1.123**, F1 **0.601**
- **Evaluación**:  

---

## Exp_4 — *convnextT_mm_20250824_195656*
- **Modelo**: ConvNeXt_Tiny (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5). 
- **Fecha**: 24/08/2025 19:56
- **Run ID**: `791938564485270177`  
- **Ejecución**: 3.4 h (CPU)  
- **Configuración**:  
  - **Transform (train)**: `RandomHorizontalFlip(p=0.2)`, `RandomRotation(15)`, normalización ImageNet.  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=2e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=6.
- **Resultados**:  
  - **Entrenamiento**: Acc **0.838**, Loss **0.786**, F1 **0.869** 
  - **Validación**: Acc **0.637**, Loss **1.606**, F1 **0.580**  
  - **Test**: Acc **0.631**, Loss **1.085**, F1 **0.608**
- **Evaluación**:  

---

## Exp_5 — *vitB16_mm_20250825_000841*
- **Modelo**: ViT_B_16 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5). 
- **Fecha**: 25/08/2025 12:08
- **Run ID**: `e45b7de5a2d84a2babf4333631e2a0b3`  
- **Ejecución**: 9.9 h (CPU)  
- **Configuración**:  
  - **Transform (train)**: `RandomHorizontalFlip(p=0.2)`, `RandomRotation(15)`, normalización ImageNet.  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=2e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=6.
- **Resultados**:  
  - **Entrenamiento**: Acc **0.508**, Loss **1.240**, F1 **0.553** 
  - **Validación**: Acc **0.437**, Loss **1.677**, F1 **0.407**  
  - **Test**: Acc **0.433**, Loss **1.207**, F1 **0.477**
- **Evaluación**:  

---

## Exp_6 — *resnet18_mm_20250825_170002*
- **Modelo**: ResNet18 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5). 
- **Fecha**: 25/08/2025 17:00
- **Run ID**: `af1528b0a851483e9165753458074fed`  
- **Ejecución**: 1.3 h (CPU)  
- **Configuración**:  
  - **Transform (train)**: `RandomHorizontalFlip(p=0.2)`, `RandomRotation(15)`, normalización ImageNet.  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=2e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=6.
- **Resultados**:  
  - **Entrenamiento**: Acc **0.804**, Loss **0.814**, F1 **0.837** 
  - **Validación**: Acc **0.547**, Loss **1.765**, F1 **0.487**  
  - **Test**: Acc **0.582**, Loss **1.146**, F1 **0.539**
- **Evaluación**:  

---

## Tabla resumen de métricas

| Exp | Fecha      | #Clases | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 | Test Prec | Test Rec |
| --: | ---------- | :-----: | :-------: | :------: | :-----: | :----: | :------: | :-----: | :-------: | :------: |
|   1 | 25/08/2025 |    8    |   0.864   |   0.882  |  0.610  |  0.518 |   0.644  |  0.587  |   0.600   |   0.596  |
|   2 | 24/08/2025 |    8    |   0.808   |   0.835  |  0.547  |  0.525 |   0.608  |  0.573  |   0.544   |   0.617  |
|   3 | 24/08/2025 |    8    |   0.781   |   0.816  |  0.599  |  0.552 |   0.619  |  0.601  |   0.595   |   0.626  |
|   4 | 24/08/2025 |    8    |   0.838   |   0.869  |  0.637  |  0.580 |   0.631  |  0.608  |   0.665   |   0.605  |
|   5 | 25/08/2025 |    8    |   0.508   |   0.553  |  0.437  |  0.407 |   0.433  |  0.477  |   0.466   |   0.561  |
|   6 | 25/08/2025 |    8    |   0.804   |   0.837  |  0.547  |  0.487 |   0.582  |  0.539  |   0.511   |   0.586  |

---

## Conclusiones

-   
- 
- 

---

## Exp_3_2 — *efficientnetb0_mm_20250825_182828*
- **Modelo**: EfficientNet_B0 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5). 
- **Fecha**: 25/08/2025 18:28
- **Run ID**: `e949df22a6c846ca8ca8075c72230f74`  
- **Ejecución**: 1.1 h (CPU)  
- **Cambios respecto a Exp_3**:  
  - `weight_decay` **2e-4→5e-4** (más regularización).  
  - 'patience' = 6→4
- **Configuración**:  
  - **Transform (train)**: `RandomHorizontalFlip(p=0.2)`, `RandomRotation(15)`, normalización ImageNet.  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=5e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=4.
- **Resultados**:  
  - **Entrenamiento**: Acc **0.781**, Loss **0.835**, F1 **0.815** 
  - **Validación**: Acc **0.601**, Loss **1.592**, F1 **0.547**  
  - **Test**: Acc **0.636**, Loss **1.115**, F1 **0.613**
- **Evaluación**:  

---

## Exp_3_3 — *efficientnetb0_mm_20250826_120831*
- **Modelo**: EfficientNet_B0 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5). 
- **Fecha**: 26/08/2025 12:08
- **Run ID**: `5246858b7cc84187840473e6fbcca541`  
- **Ejecución**: 1.0 h (CPU)  
- **Cambios respecto a Exp_3**:  
  - `weight_decay` **2e-4→5e-4** (más regularización).  
  - 'patience' = 6→4
  - **augmentation** enriquecido: `RandomAffine(±7°, ±2% shift, 0.95–1.05 scale)` + `ColorJitter(0.10/0.10/0.10/0.02)` + `Flip(0.5)` + `RandomErasing(0.25)`.
- **Configuración**:  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=5e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=4.
- **Resultados**:  
  - **Entrenamiento**: Acc **0.759**, Loss **0.842**, F1 **0.799** 
  - **Validación**: Acc **0.586**, Loss **1.612**, F1 **0.532**  
  - **Test**: Acc **0.636**, Loss **1.097**, F1 **0.591**
- **Evaluación**:  

---

## Exp_3_4 — *efficientnetb0_mm_subclases20250826_162142*
- **Modelo**: EfficientNet_B0 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5). 
- **Fecha**: 26/08/2025 16:21
- **Run ID**: `0d568c86822f43408228831469641d34`  
- **Ejecución**: 53.8 min (CPU)  
- **Cambios respecto a Exp_3**: 
  - Se **reintroducen metadatos**.  
  - **Reducción de clases** a **[0,1,2,5,6]** → `num_classes=5`. 
  - `weight_decay` **2e-4→5e-4** (más regularización).  
  - 'patience' = 6→4
  - **augmentation** enriquecido: `RandomAffine(±7°, ±2% shift, 0.95–1.05 scale)` + `ColorJitter(0.10/0.10/0.10/0.02)` + `Flip(0.5)` + `RandomErasing(0.25)`.
- **Configuración**:  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=5e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=4.
- **Resultados**:  
  - **Entrenamiento**: Acc **0.882**, Loss **0.661**, F1 **0.909** 
  - **Validación**: Acc **0.706**, Loss **1.040**, F1 **0.724**  
  - **Test**: Acc **0.754**, Loss **0.645**, F1 **0.777**
- **Evaluación**:  

---

## Exp_3_5— *efficientnetb0_mm_subclases_sin_augmentation20250826_175741*
- **Modelo**: EfficientNet_B0 (imagen) + MLP (metadatos, `meta_dim=2`), fusión por concatenación y cabeza MLP (256 + dropout=0.5). 
- **Fecha**: 26/08/2025 16:21
- **Run ID**: `0d568c86822f43408228831469641d34`  
- **Ejecución**: 53.8 min (CPU)  
- **Cambios respecto a Exp_3**: 
  - Se **reintroducen metadatos**.  
  - **Reducción de clases** a **[0,1,2,5,6]** → `num_classes=5`. 
  - `weight_decay` **2e-4→5e-4** (más regularización).  
  - 'patience' = 6→4
- **Configuración**:  
  - **Transform (train)**: `RandomHorizontalFlip(p=0.5)`, `RandomRotation(15)`, normalización ImageNet.  
  - **Parámetros**: `epochs=15`, `batch_size=32`, `lr=1e-4`, `weight_decay=5e-4`, `num_workers=0`, `num_classes=8`. 
  - **Criterio de selección**: **F1-macro** en validación; **Early Stopping** paciencia=4.
- **Resultados**:  
  - **Entrenamiento**: Acc **0.958**, Loss **0.579**, F1 **0.967** 
  - **Validación**: Acc **0.727**, Loss **1.173**, F1 **0.736**  
  - **Test**: Acc **0.752**, Loss **0.696**, F1 **0.772**
- **Evaluación**:  

---


## Tabla resumen de métricas

| Exp | Fecha      | #Clases | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 | Test Prec | Test Rec |
| --: | ---------- | :-----: | :-------: | :------: | :-----: | :----: | :------: | :-----: | :-------: | :------: |
|   3 | 24/08/2025 |    8    |   0.781   |   0.816  |  0.599  |  0.552 |   0.619  |  0.601  |   0.595   |   0.626  |
| 3_2 | 25/08/2025 |    8    |   0.781   |   0.815  |  0.601  |  0.547 |   0.636  |  0.613  |   0.602   |   0.637  |
| 3_3 | 26/08/2025 |    8    |   0.759   |   0.799  |  0.586  |  0.532 |   0.636  |  0.591  |   0.590   |   0.619  |
| 3_4 | 26/08/2025 |    5    |   0.882   |   0.909  |  0.706  |  0.724 |   0.754  |  0.777  |   0.771   |   0.785  |
| 3_5 | 26/08/2025 |    5    |   0.958   |   0.967  |  0.727  |  0.736 |   0.752  |  0.772  |   0.785   |   0.764  |


---

## Hiperarámetros principales 

| Exp | Batch | Epochs | LR inicial | Weight Decay | Augmentación principal                     | Dropout | Scheduler | Optimizer |
| --: | :---: | :----: | :--------: | :----------: | ------------------------------------------ | :-----: | :-------: | :-------: |
|   3 |   32  |   15   |    1e-4    |     2e-4     | Flip(0.2)+Rot(15)                          |   0.5   |     No    |   AdamW   |
| 3_2 |   32  |   15   |    1e-4    |     5e-4     | Flip(0.2)+Rot(15)                          |   0.5   |     No    |   AdamW   |
| 3_3 |   32  |   15   |    1e-4    |     5e-4     | Affine+ColorJitter+Flip(0.5)+Erasing(0.25) |   0.5   |     No    |   AdamW   |
| 3_4 |   32  |   15   |    1e-4    |     5e-4     | Affine+ColorJitter+Flip(0.5)+Erasing(0.25) |   0.5   |     No    |   AdamW   |
| 3_5 |   32  |   15   |    1e-4    |     5e-4     | Flip(0.5)+Rot(15)                          |   0.5   |     No    |   AdamW   |


---

## Conclusiones

-  
- 
- 




