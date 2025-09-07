#!/usr/bin/env python3
"""
transform.py – versión OCR segura para Windows

Uso:
    python transform.py documento.pdf

Funcionalidades:
- Limpieza de texto (líneas de índice, dot-leaders, repeticiones)
- Detecta PDFs escaneados y aplica OCR automáticamente
- Usa PyMuPDF + pytesseract (sin Poppler)
- Permite asignar ruta manual de Tesseract si no está en PATH
- Guarda PDF original y .txt limpio en Fixed/
"""

from __future__ import annotations
import argparse
import pathlib
import re
import shutil
import unicodedata
from collections import Counter
from typing import List

try:
    from pdfminer.high_level import extract_text
except ImportError:
    raise SystemExit("❌ pdfminer.six no instalado. pip install pdfminer.six")

try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
except ImportError:
    raise SystemExit("❌ PyMuPDF, pytesseract o pillow no instalados. pip install PyMuPDF pytesseract pillow")

# --------------------- Configuración ---------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
FIXED_DIR = BASE_DIR / "Fixed"
FIXED_DIR.mkdir(parents=True, exist_ok=True)

# Si Tesseract no está en PATH, asigna la ruta completa aquí:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------- Limpieza ---------------------
DOT_LEADER_RE = re.compile(r"[.•⋅·]{3,}")
INDEX_LINE_RE = re.compile(r"[.•⋅·]{3,}\s*\d+\s*$")
REPEATED_INLINE_RE = re.compile(r"\b(\w{2,5})(?:\1){2,}\b", re.IGNORECASE)
CONSEC_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)

def collapse_repeated_words(text: str) -> str:
    text = CONSEC_WORD_RE.sub(r"\1", text)
    text = REPEATED_INLINE_RE.sub(r"\1", text)
    return text

def join_paragraph_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    buffer: List[str] = []

    def flush():
        if buffer:
            out.append(" ".join(buffer))
            buffer.clear()

    for ln in lines:
        if ln:
            buffer.append(ln)
        else:
            flush()
    flush()
    return out

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    raw_lines = [ln.rstrip() for ln in text.splitlines()]
    candidate_lines: List[str] = []
    for ln in raw_lines:
        if not ln.strip():
            continue
        if INDEX_LINE_RE.search(ln):
            continue
        ln = DOT_LEADER_RE.sub(" ", ln)
        candidate_lines.append(ln.strip())

    freq = Counter(candidate_lines)
    approx_pages = max(1, len(candidate_lines) // 40)

    filtered: List[str] = []
    prev = ""
    for ln in candidate_lines:
        if ln.isdigit():
            continue
        if freq[ln] > 0.8 * approx_pages and len(ln.split()) < 10:
            continue
        if ln == prev:
            continue
        filtered.append(ln)
        prev = ln

    paragraphs = join_paragraph_lines(filtered)
    cleaned = "\n\n".join(paragraphs)
    cleaned = collapse_repeated_words(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()

# --------------------- PDF ---------------------
def is_scanned_pdf(pdf_path: pathlib.Path) -> bool:
    try:
        sample = extract_text(str(pdf_path), maxpages=2)
        return len(sample.strip()) < 30
    except Exception:
        return True

def extract_text_pdf(pdf_path: pathlib.Path) -> str:
    return extract_text(str(pdf_path))

def ocr_pdf(pdf_path: pathlib.Path) -> str:
    doc = fitz.open(str(pdf_path))
    text = ""
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            page_text = pytesseract.image_to_string(img, lang="spa")
        except pytesseract.pytesseract.TesseractNotFoundError:
            raise SystemExit("❌ Tesseract no encontrado. Instálalo y actualiza pytesseract.pytesseract.tesseract_cmd")
        text += page_text + "\n"
    return text

# --------------------- Main ---------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae y limpia texto de un PDF → Fixed/")
    parser.add_argument("pdf_file", type=pathlib.Path, help="Nombre o ruta relativa del PDF")
    args = parser.parse_args()
    pdf_path = args.pdf_file

    if not pdf_path.exists():
        raise SystemExit(f"❌ Archivo '{pdf_path}' no encontrado")

    print(f"➡️ Procesando: {pdf_path.name}")
    if is_scanned_pdf(pdf_path):
        print("🤖 PDF escaneado detectado, aplicando OCR...")
        raw = ocr_pdf(pdf_path)
    else:
        print("📖 PDF normal detectado, extrayendo texto...")
        raw = extract_text_pdf(pdf_path)

    cleaned = clean_text(raw)

    # Guardar
    shutil.copy2(pdf_path, FIXED_DIR / pdf_path.name)
    txt_path = FIXED_DIR / f"{pdf_path.stem}.txt"
    txt_path.write_text(cleaned, encoding="utf-8")
    print(f"🎉 Procesado con éxito. Texto limpio guardado en: {txt_path}")

if __name__ == "__main__":
    main()


