# General/transform_imgV2.py
import sys
import argparse
from pathlib import Path

import cv2
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from fun_transformV2 import square_image


def main():
    parser = argparse.ArgumentParser(
        description="Recorta a contenido, hace cuadrado y redimensiona imágenes."
    )
    parser.add_argument("size", type=int, help="Tamaño de salida (ej. 224 o 300)")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Carpeta de entrada (por defecto: <repo>/ODIR-5K/ODIR-5K/Training Images)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Carpeta de salida (por defecto: <repo>/<size>x<size>)",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".jpg,.jpeg,.png,.tif,.tiff,.bmp",
        help="Extensiones válidas separadas por coma",
    )
    parser.add_argument("--recursive", action="store_true", help="Buscar recursivamente")
    parser.add_argument(
        "--overwrite", action="store_true", help="Sobrescribir si ya existe"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Umbral luminancia para detectar contenido",
    )
    parser.add_argument(
        "--margin", type=int, default=8, help="Margen (px) alrededor del contenido"
    )
    args = parser.parse_args()

    size = args.size
    if size <= 0:
        print("El tamaño debe ser un entero positivo.")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    input_dir = (
        Path(args.input)
        if args.input
        else (repo_root / "ODIR-5K" / "ODIR-5K" / "Training Images")
    )
    output_dir = (
        Path(args.output) if args.output else (repo_root / f"{size}x{size}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Entrada :", input_dir, "| existe?", input_dir.exists())
    print("Salida  :", output_dir)

    if not input_dir.exists():
        print(f"ERROR: Carpeta de entrada no encontrada: {input_dir}")
        sys.exit(1)

    valid_exts = tuple(
        e.strip().lower() for e in args.exts.split(",") if e.strip()
    )

    # Listado de imágenes
    if args.recursive:
        files = [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_exts
        ]
    else:
        files = [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]
    files.sort()

    print(f"Imágenes encontradas: {len(files)}")
    if files[:5]:
        print("Ejemplos:", [p.name for p in files[:5]])

    transform_resize = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                (size, size),
                interpolation=InterpolationMode.LANCZOS,
                antialias=True,
            ),
        ]
    )

    processed = skipped = failed = 0
    for src in tqdm(files, desc="Procesando", unit="img"):
        dst = output_dir / src.name

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            img_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img_bgr is None:
                failed += 1
                if failed <= 5:
                    print(f"\n[WARN] No se pudo leer: {src}")
                continue

            img_bgr_sq = square_image(
                img_bgr,
                threshold=args.threshold,
                margin=args.margin,
                pad_mode="constant",
                pad_value=0,
            )
            img_rgb_sq = img_bgr_sq[:, :, ::-1]

            pil_img = transform_resize(img_rgb_sq)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            ext = dst.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                pil_img.save(dst, format="JPEG", quality=95, optimize=True)
            elif ext == ".png":
                pil_img.save(dst, format="PNG", optimize=True)
            else:
                pil_img.save(dst)

            processed += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"\n[WARN] Falló '{src.name}': {e}")

    print("\n--- Resumen ---")
    print(f"Procesadas: {processed}")
    print(f"Saltadas : {skipped} (existían y no --overwrite)")
    print(f"Fallidas : {failed}")
    print("Transformación completada.")


if __name__ == "__main__":
    main()
