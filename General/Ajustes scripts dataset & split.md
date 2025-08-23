
## Ajustes en eye_pytorch_dataset.py

eye_pytorch_dataset.py: comenta “multi-label only”. Este sería el caso de haber elegido 'target' como y.

eye_pytorch_dataset_Leti.py: se corrije a multiclase (tenemos un solo label por ojo).

----

He cambiadolas rutas para que se pueda usar con cualquier sistema operativo sin que de error, más portable. Uso pathlib.Path.


----

Transforms:

He añadido get_train_transform(img_size=224) y get_eval_transform(img_size=224) con RandomHorizontalFlip(p=0.2) y RandomRotation(15), para que sea más modular y facil de ajustar.

----

Apertura de imágenes:

Image.open().convert("RGB"), lo he cambiado a: with Image.open() as im

De esta manera aseguramos que cada archivo de imagen se cierra correctamente, que no se acumulen muchos archivos abiertos.

----

Esto todvía no lo he cambiado!! Parámetro num_classes no se usa. O lo eliminas o podemos validarlo con un assert cod_target < num_classes. 


------



## create_split_dataset.py:


Cambio a pathlib.Path, resuelve repo_root y crea Data/dataset/ si no existe: portable y robusto.

----

Transforms:

"""
dataset_train = Subset(ds, tr_idx)
dataset_train.transform = train_imgs_transforms
"""

no se transforman las imágenes. Subset solo guarda una lista de índices y reenvía la llamada al dataset original. No tiene lógica para aplicar transform.

He corregido para que use get_train_transform(224) y get_eval_transform(224) (importados del dataset).

----

Construcción de datasets y guardado:

Creo EyeDataset y luego Subset para train/val/test, guardando directamente los Subsets con torch.save() → listos para torch.load sin necesidad de reconstruirlos.

Antes en los archivos .pt estaba guardando diccionarios. Al cargar para el dataloader habría que reconstruir el Subset manualmente, DataLoader espera un Dataset o Subset, no un diccionario.







