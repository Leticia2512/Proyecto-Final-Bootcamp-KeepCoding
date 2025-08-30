''' from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import unicodedata
from collections import Counter
from typing import Any, List

#pymupdf

try:
    from pdfminer.high_level import extract_text  # type: ignore
except ImportError as e:
    raise SystemExit() from e

# -------------------------------- Path config -------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
FIXED_DIR = BASE_DIR / "Fixed_david"      #OJO CAMBIADO PARA ENVIAR A UNA CARPETA NUEVA "DAVID"
FIXED_DIR.mkdir(parents=True, exist_ok=True)

print("➡️ Script cargado. BASE_DIR:", BASE_DIR)
print("➡️ Guardando en FIXED_DIR:", FIXED_DIR)

# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------

def is_scanned_pdf(pdf_path: pathlib.Path) -> bool:
    """Heurística muy simple: si pdfminer devuelve <30 caracteres en 2 páginas ➜ escaneado."""
    print("🔍 Comprobando si el PDF está escaneado:", pdf_path)
    try:
        sample = extract_text(str(pdf_path), maxpages=2)
        print("   → Caracteres extraídos en las 2 primeras páginas:", len(sample.strip()))
        return len(sample.strip()) < 30
    except Exception as e:
        print("   ⚠️ Error al extraer texto con pdfminer:", e)
        return True

def extract_text_pdf(pdf_path: pathlib.Path) -> str:
    print("📖 Extrayendo texto completo con pdfminer...")
    return extract_text(str(pdf_path))

# ---------------------------------------------------------------------------
# LIMPIEZA
# ---------------------------------------------------------------------------

DOT_LEADER_RE = re.compile(r"[.•⋅·]{3,}")
INDEX_LINE_RE = re.compile(r"[.•⋅·]{3,}\s*\d+\s*$")  # línea de índice «……  12»
REPEATED_INLINE_RE = re.compile(
    r"\b(\w{2,5})(?:\1){2,}\b", re.IGNORECASE)  # gpcgpcgpc
CONSEC_WORD_RE = re.compile(
    r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)  # gpc gpc gpc

def collapse_repeated_words(text: str) -> str:
    return CONSEC_WORD_RE.sub(r"\1", text).strip()

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
    print("🧹 Limpiando texto extraído...")
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

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("➡️ Entrando en main()")
    parser = argparse.ArgumentParser(description="Extrae y limpia texto de un PDF → RAG/Fixed_david/.")
    parser.add_argument("pdf_file", type=pathlib.Path, help="Nombre o ruta relativa del PDF")
    args = parser.parse_args()
    pdf_path = args.pdf_file
    print("➡️ PDF recibido como argumento:", pdf_path)

    if not pdf_path.exists():
        print("❌ El archivo no existe en esa ruta:", pdf_path)
        raise SystemExit(f"❌ Archivo '{pdf_path}' no encontrado")

    if is_scanned_pdf(pdf_path):
        print("⚠️ PDF parece escaneado; se copia tal cual (añadir OCR en futuras versiones).")
        shutil.copy2(pdf_path, FIXED_DIR / pdf_path.name)
        print("✅ Copiado en:", FIXED_DIR / pdf_path.name)
        return

    raw = extract_text_pdf(pdf_path)
    print("📏 Longitud del texto extraído:", len(raw))

    cleaned = clean_text(raw)
    print("📏 Longitud del texto limpio:", len(cleaned))

    shutil.copy2(pdf_path, FIXED_DIR / pdf_path.name)
    txt_path = FIXED_DIR / f"{pdf_path.stem}.txt"
    txt_path.write_text(cleaned, encoding="utf-8")

    print("✅ Procesado: texto limpio guardado en", txt_path)

if __name__ == "__main__":
    main() '''

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import unicodedata
from collections import Counter
from typing import Any, List

try:
    from pdfminer.high_level import extract_text  # type: ignore
except ImportError as e:
    raise SystemExit() from e

# -------------------------------- Path config -------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
FIXED_DIR = BASE_DIR / "Fixed_david"      # Carpeta propia para tus txt limpios
FIXED_DIR.mkdir(parents=True, exist_ok=True)

print("➡️ Script cargado. BASE_DIR:", BASE_DIR)
print("➡️ Guardando en FIXED_DIR:", FIXED_DIR)

# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------

def is_scanned_pdf(pdf_path: pathlib.Path) -> bool:
    """Heurística muy simple: si pdfminer devuelve <30 caracteres en 2 páginas ➜ escaneado."""
    print("🔍 Comprobando si el PDF está escaneado:", pdf_path)
    try:
        sample = extract_text(str(pdf_path), maxpages=2)
        print("   → Caracteres extraídos en las 2 primeras páginas:", len(sample.strip()))
        return len(sample.strip()) < 30
    except Exception as e:
        print("   ⚠️ Error al extraer texto con pdfminer:", e)
        return True

def extract_text_pdf(pdf_path: pathlib.Path) -> str:
    print("📖 Extrayendo texto completo con pdfminer...")
    return extract_text(str(pdf_path))

# ---------------------------------------------------------------------------
# LIMPIEZA
# ---------------------------------------------------------------------------

DOT_LEADER_RE = re.compile(r"[.•⋅·]{3,}")
INDEX_LINE_RE = re.compile(r"[.•⋅·]{3,}\s*\d+\s*$")
REPEATED_INLINE_RE = re.compile(r"\b(\w{2,5})(?:\1){2,}\b", re.IGNORECASE)
CONSEC_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)

def collapse_repeated_words(text: str) -> str:
    return CONSEC_WORD_RE.sub(r"\1", text).strip()

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
    print("🧹 Limpiando texto extraído...")
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

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("➡️ Entrando en main()")
    parser = argparse.ArgumentParser(description="Extrae y limpia texto de un PDF → RAG/Fixed_david/.")
    parser.add_argument("pdf_file", type=pathlib.Path, nargs="?", help="Nombre o ruta relativa del PDF")
    args = parser.parse_args()

    # PDF por defecto si no se pasa argumento
    if args.pdf_file is None:
        pdf_path = BASE_DIR / "Guia_SERV_01_segundaRevision.pdf"
        print(f"⚠️ No se indicó PDF. Usando por defecto: {pdf_path}")
    else:
        pdf_path = args.pdf_file

    print("➡️ PDF recibido:", pdf_path)

    if not pdf_path.exists():
        print("❌ El archivo no existe en esa ruta:", pdf_path)
        raise SystemExit(f"❌ Archivo '{pdf_path}' no encontrado")

    if is_scanned_pdf(pdf_path):
        print("⚠️ PDF parece escaneado; se copia tal cual (añadir OCR en futuras versiones).")
        shutil.copy2(pdf_path, FIXED_DIR / pdf_path.name)
        print("✅ Copiado en:", FIXED_DIR / pdf_path.name)
        return

    raw = extract_text_pdf(pdf_path)
    print("📏 Longitud del texto extraído:", len(raw))

    cleaned = clean_text(raw)
    print("📏 Longitud del texto limpio:", len(cleaned))

    shutil.copy2(pdf_path, FIXED_DIR / pdf_path.name)
    txt_path = FIXED_DIR / f"{pdf_path.stem}.txt"
    txt_path.write_text(cleaned, encoding="utf-8")

    print("✅ Procesado: texto limpio guardado en", txt_path)

if __name__ == "__main__":
    main()


