import os
import sys
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from fun_transform import square_image
import cv2

# --- Comprobar argumentos ---
if len(sys.argv) != 2:
    print("Uso: python transform_img.py <tamaño>")
    sys.exit(1)

try:
    size = int(sys.argv[1])
except ValueError:
    print("Error: el tamaño debe ser un número entero.")
    sys.exit(1)

# --- Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
input_dir = os.path.join(parent_dir, "ODIR-5K", "ODIR-5K", "Training Images")

output_dir = os.path.join(parent_dir, f"{size}x{size}")

os.makedirs(output_dir, exist_ok=True)

# --- Transformación ---
transform_resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size)
])

# Extensiones válidas
valid_ext = ".jpg"

# Lista de imágenes
images = [f for f in os.listdir(input_dir) if os.path.splitext(f)[
    1].lower() in valid_ext]

print(f"Total de imágenes encontradas: {len(images)}")
print(f"Transformando a tamaño {size}x{size}...")

# --- Proceso ---
for fname in tqdm(images, desc="Procesando", unit="img"):
    src = os.path.join(input_dir, fname)
    dst = os.path.join(output_dir, fname)

    img = cv2.imread(src)
    img = square_image(img)
    img = transform_resize(img[:,:,::-1])

    # Asegurar que la imagen sea cuadrada
    img.save(dst, quality=95)

print(f" Transformación completada.")
