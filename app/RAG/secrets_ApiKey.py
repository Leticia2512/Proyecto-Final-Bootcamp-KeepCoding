# secrets.py
from pathlib import Path
import os

# Nombre de fichero local para la API key( OJO NO SUBIR A GIT)
KEY_FILENAME = ".openai_key"
KEY_PATH = Path(KEY_FILENAME)


def get_openai_api_key(required: bool = True) -> str | None:
    """
    Devuelve la OpenAI API key.
    - Primero comprueba la variable de entorno OPENAI_API_KEY.
    - Si no existe, lee el fichero local .openai_key (en la raíz del repo).
    - Si no encuentra y required=True lanza RuntimeError.
    """
    # 1) variable de entorno
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()

    # 2) fichero .openai_key (local, en la raíz del repo)
    if KEY_PATH.exists():
        key_text = KEY_PATH.read_text(encoding="utf-8").strip()
        if key_text:
            return key_text

    if required:
        raise RuntimeError(
            "OpenAI API key no encontrada. Coloca tu key en la variable de entorno "
            "'OPENAI_API_KEY' o crea un fichero '.openai_key' en la raíz del proyecto "
            "con la key en una sola línea. Asegúrate de añadir '.openai_key' a .gitignore."
        )
    return None


if __name__ == "__main__":
    try:
        print("OpenAI key encontrada:", bool(get_openai_api_key()))
    except Exception as e:
        print("Error:", e)
