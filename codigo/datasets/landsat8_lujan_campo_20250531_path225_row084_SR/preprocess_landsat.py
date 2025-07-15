import os
import numpy as np
import rasterio
from pathlib import Path


def leer_banda(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile

def apilar_bandas(directorio, nombres_bandas):
    rutas = [Path(directorio) / nombre for nombre in nombres_bandas]

    # Verificación explícita
    for ruta in rutas:
        if not ruta.exists():
            raise FileNotFoundError(f"❌ No se encontró la banda: {ruta}")

    bandas = [rasterio.open(str(ruta)).read(1) for ruta in rutas]
    return np.stack(bandas, axis=-1)  # shape: (alto, ancho, canales)

def leer_mascara_etiquetas(path_mascara):
    with rasterio.open(str(path_mascara)) as src:
        return src.read(1).astype(np.uint8)  # etiquetas por píxel

def generar_dataset_por_pixel(imagen_stack, mascara):
    alto, ancho, canales = imagen_stack.shape
    X = imagen_stack.reshape(-1, canales)
    y = mascara.flatten()

    # Filtramos los píxeles sin etiqueta (por ejemplo, valor 0 si es fondo)
    mascara_valida = y > 0
    X_filtrado = X[mascara_valida]
    y_filtrado = y[mascara_valida]

    return X_filtrado, y_filtrado


def detectar_bandas_sr(path_dir):
    path_dir = Path(path_dir)
    bandas = []
    for i in range(2, 8):  # B2 a B7
        encontrados = list(path_dir.glob(f"*SR_B{i}.TIF"))
        if not encontrados:
            raise FileNotFoundError(f"No se encontró la banda SR_B{i} en {path_dir}")
        bandas.append(encontrados[0].name)
    return bandas

def cargar_dataset_landsat(path_dir, path_mascara):
    # Selección de bandas reflectancia de superficie (ej: bandas 2 a 7)
    nombres_bandas = detectar_bandas_sr(path_dir)

    imagen = apilar_bandas(path_dir, nombres_bandas)
    mascara = leer_mascara_etiquetas(path_mascara)
    X, y = generar_dataset_por_pixel(imagen, mascara)
    return X, y
