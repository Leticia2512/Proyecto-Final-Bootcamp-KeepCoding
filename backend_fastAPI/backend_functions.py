from pathlib import Path

def validate_image_jpg(ruta_imagen: str) -> bool:
    """
    Valida si una ruta de archivo corresponde a una imagen JPG y si existe.

    Args:
        ruta_imagen: La ruta de la imagen como una cadena de texto.

    Returns:
        True si la ruta es válida y el archivo existe, False en caso contrario.
    """
    ruta = Path(ruta_imagen)

    url_status = "La url es correcta."
    
    # Confirmamos que termine con una extesión .jpg
    if not ruta.suffix.lower() == '.jpg':
        url_status = f"Error: La extensión del archivo no es .jpg. La extensión es: {ruta.suffix}. Debes introducir una imágen tipo JPG"
        return False, url_status
    
    # Revisa si el archivo existe
    if not ruta.exists():
        url_status = f"Error: El archivo no existe en la ruta especificada: {ruta}"
        return False, url_status
        
    return True, url_status
 
