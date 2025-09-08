
"""


Uso:
    python transform.py documento.pdf

Cambios clave v0.3
------------------
* **Elimina líneas del índice** (líneas con "……" + nº de página).
* **Borra o limpia los "dot-leaders"** (······) dentro de una línea.
* **Detecta repeticiones dentro de una misma palabra** como «gpcgpcgpc» y las colapsa.
* Mejora `collapse_repeated_words` para manejar tokens pegados.

Restante pipeline: carga, heurística de PDF escaneado, extracción con pdfminer, limpieza avanzada y guardado en `RAG/Fixed/`.
"""

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
FIXED_DIR = BASE_DIR / "fixed"
FIXED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------


def is_scanned_pdf(pdf_path: pathlib.Path,
                   lookahead_pages: int = 10,
                   min_letters_threshold: int = 120,
                   blank_page_letters: int = 10,
                   consider_pages_after_blanks: int = 5) -> bool:
    """
    Heurística robusta:
    - Mira hasta `lookahead_pages`.
    - Ignora portadas en blanco (páginas con < `blank_page_letters` letras).
    - Suma letras de las siguientes `consider_pages_after_blanks` páginas.
    - Si la suma < `min_letters_threshold` => parece escaneado.
    """
    try:
        sample = extract_text(str(pdf_path), maxpages=lookahead_pages) or ""
        # pdfminer separa páginas con \f (form feed)
        pages = sample.split("\f")

        # cuenta letras por página (más robusto que len total)
        letters_per_page = [sum(ch.isalpha() for ch in pg) for pg in pages]

        # salta portadas/páginas iniciales prácticamente vacías
        i = 0
        while i < len(letters_per_page) and letters_per_page[i] < blank_page_letters:
            i += 1

        # suma de las siguientes N páginas "reales"
        window = letters_per_page[i:i + consider_pages_after_blanks]
        total_letters = sum(window)

        # DEBUG opcional:
        # print("DEBUG letters_per_page:", letters_per_page)
        # print("DEBUG skipped:", i, "window:", window, "total_letters:", total_letters)

        return total_letters < min_letters_threshold
    except Exception:
        # Si pdfminer peta, mantenemos el comportamiento conservador
        return True

# ---------------------------------------------------------------------------
# LIMPIEZA
# ---------------------------------------------------------------------------


# tres o más puntos / puntos medios etc.
DOT_LEADER_RE = re.compile(r"[.•⋅·]{3,}")
INDEX_LINE_RE = re.compile(r"[.•⋅·]{3,}\s*\d+\s*$")  # línea de índice «……  12»
REPEATED_INLINE_RE = re.compile(
    r"\b(\w{2,5})(?:\1){2,}\b", re.IGNORECASE)  # gpcgpcgpc
CONSEC_WORD_RE = re.compile(
    r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)  # gpc gpc gpc


def collapse_repeated_words(text: str) -> str:
    """Colapsa repeticiones tanto *separadas por espacio* como *pegadas* en la misma palabra."""
    text = CONSEC_WORD_RE.sub(r"\1", text)
    text = REPEATED_INLINE_RE.sub(r"\1", text)
    return text


def join_paragraph_lines(lines: List[str]) -> List[str]:
    """Une líneas consecutivas no vacías en párrafos; deja doble salto entre párrafos."""
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
    """Proceso de limpieza principal."""

    text = unicodedata.normalize("NFC", text)

    # 1. separa líneas, strip de espacios finales
    raw_lines = [ln.rstrip() for ln in text.splitlines()]

    # 2. descarta totalmente líneas-índice (…… 12) o vacías
    candidate_lines: List[str] = []
    for ln in raw_lines:
        if not ln.strip():
            continue
        if INDEX_LINE_RE.search(ln):
            continue
        # elimina dot-leaders dentro de la línea (pero mantiene título)
        ln = DOT_LEADER_RE.sub(" ", ln)
        candidate_lines.append(ln.strip())

    # 3. frecuencias → cabeceras/pies
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

    # 4. fusiona líneas en párrafos
    paragraphs = join_paragraph_lines(filtered)
    cleaned = "\n\n".join(paragraphs)

    # 5. colapsa repeticiones
    cleaned = collapse_repeated_words(cleaned)

    # 6. normaliza espacios y saltos
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae y limpia texto de un PDF → RAG/Fixed/.")
    parser.add_argument("pdf_file", type=pathlib.Path,
                        help="Nombre o ruta relativa del PDF")
    args = parser.parse_args()
    pdf_path = args.pdf_file

    if not pdf_path.exists():
        raise SystemExit(f"Archivo '{pdf_path}' no encontrado")

    if is_scanned_pdf(pdf_path):
        print(
            "PDF parece escaneado; se copia tal cual (añadir OCR en futuras versiones).")
        shutil.copy2(pdf_path, FIXED_DIR / pdf_path.name)
        return

    raw = extract_text(pdf_path)
    cleaned = clean_text(raw)

    # Copia original + txt limpio
    shutil.copy2(pdf_path, FIXED_DIR / pdf_path.name)
    txt_path = FIXED_DIR / f"{pdf_path.stem}.txt"
    txt_path.write_text(cleaned, encoding="utf-8")

    print("Procesado: texto limpio en", txt_path)


if __name__ == "__main__":
    main()
