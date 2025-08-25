# fun_transformV2.py

from __future__ import annotations
import numpy as np
from typing import Tuple, Literal, Optional

PadMode = Literal["constant", "edge", "reflect"]

def _ensure_3d(img: np.ndarray) -> np.ndarray:
    """
    Asegura que la imagen tenga 3 dimensiones (H, W, C).
    - Si es HxW, añade canal ficticio.
    - Si es HxWx4 (RGBA), descarta alpha y deja HxWx3.
    - Si es HxWx1, lo deja tal cual en 3D.
    """
    if img.ndim == 2:
        img = img[..., None]  # HxW -> HxWx1
    if img.ndim != 3:
        raise ValueError(f"Se esperaba imagen 2D o 3D, recibido shape={img.shape}")
    if img.shape[2] == 4:       # RGBA -> RGB
        img = img[..., :3]
    return img

def _luminance_mask(img: np.ndarray, threshold: int | float = 2) -> np.ndarray:
    """
    Calcula máscara de contenido (True donde hay señal).
    threshold: valor en [0..255] aprox. Si la imagen es float [0..1],
               se reescala internamente a 0..255 para comparar.
    La máscara se basa en la luminancia Y ≈ 0.299R + 0.587G + 0.114B.
    Para imágenes monocanal, usa su único canal.
    """
    img3 = _ensure_3d(img)
    arr = img3.astype(np.float32)

    # Escala a 0..255 si viene normalizada [0..1]
    mx = float(arr.max()) if arr.size else 1.0
    if mx <= 1.0:
        arr = arr * 255.0

    if arr.shape[2] == 1:
        Y = arr[..., 0]
    else:
        # Luminancia aproximada
        Y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    mask = Y > float(threshold)
    return mask

def content_bbox(
    img: np.ndarray,
    threshold: int | float = 2,
    min_size: int = 1
) -> Optional[Tuple[int, int, int, int]]:
    """
    Devuelve la caja de contenido (r0, r1, c0, c1) usando máscara por luminancia.
    - threshold: píxeles por debajo se consideran fondo.
    - min_size: tamaño mínimo de alto/ancho para aceptar caja (si no, None).
    Retorna None si no se encuentra contenido.
    """
    mask = _luminance_mask(img, threshold=threshold)

    if not mask.any():
        return None

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1

    if (r1 - r0) < min_size or (c1 - c0) < min_size:
        return None
    return (r0, r1, c0, c1)

def crop_to_content(
    img: np.ndarray,
    threshold: int | float = 2,
    margin: int = 0,
    min_size: int = 1
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Recorta la imagen a su región de contenido con un margen (en píxeles) alrededor.
    Si no hay contenido, devuelve la imagen original y bbox=None.
    """
    H, W = img.shape[:2]
    bbox = content_bbox(img, threshold=threshold, min_size=min_size)
    if bbox is None:
        return img, None

    r0, r1, c0, c1 = bbox
    r0 = max(0, r0 - margin)
    c0 = max(0, c0 - margin)
    r1 = min(H, r1 + margin)
    c1 = min(W, c1 + margin)

    return img[r0:r1, c0:c1, ...], (r0, r1, c0, c1)

def pad_to_square(
    img: np.ndarray,
    pad_mode: PadMode = "constant",
    pad_value: int | float = 0
) -> np.ndarray:
    """
    Expande por padding hasta formar un cuadrado:
    - pad_mode: 'constant' (relleno constante), 'edge' (repite borde), 'reflect' (reflexión).
    - pad_value: valor del relleno si 'constant'.
    Mantiene el tipo de dato de la imagen.
    """
    H, W = img.shape[:2]
    if H == W:
        return img

    diff = abs(H - W)
    pad_before = diff // 2 + (diff % 2)
    pad_after  = diff // 2

    if H > W:
        pad_width = ((0, 0), (pad_before, pad_after))
    else:
        pad_width = ((pad_before, pad_after), (0, 0))

    # Si hay canales, añadir dimensión de no-padding para C
    if img.ndim == 3:
        pad_width = (*pad_width, (0, 0))

    if pad_mode == "constant":
        img_sq = np.pad(img, pad_width=pad_width, mode="constant", constant_values=pad_value)
    elif pad_mode in ("edge", "reflect"):
        img_sq = np.pad(img, pad_width=pad_width, mode=pad_mode)
    else:
        raise ValueError(f"pad_mode no soportado: {pad_mode}")
    return img_sq

def square_image(
    img: np.ndarray,
    threshold: int | float = 2,
    margin: int = 0,
    pad_mode: PadMode = "constant",
    pad_value: int | float = 0,
    min_size: int = 1
) -> np.ndarray:
    """
    Recorta a contenido (con margen) y expande a cuadrado por padding.
    threshold pequeño (1–5) suele ir bien en fundus con fondo casi negro.
    """
    cropped, _ = crop_to_content(img, threshold=threshold, margin=margin, min_size=min_size)
    squared = pad_to_square(cropped, pad_mode=pad_mode, pad_value=pad_value)
    return squared
