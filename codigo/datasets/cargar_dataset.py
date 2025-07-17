import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from cargar_eurosat import cargar_dataset_eurosat  

def cargar_dataset(path, clase_minoria=None, col_features=None, col_target=None,
                   sep=' ', header=None, binarizar=True, tipo='tabular'):
    if tipo == 'imagen':
        # Carga especial para datasets de imágenes como EuroSAT
        X, y, clases = cargar_dataset_eurosat(path)
        return X, y, clases

    # Carga tabular por defecto
    df = pd.read_csv(path, header=header, sep=sep)
    df = df[~df.isin(['?']).any(axis=1)]

    if col_target is None or col_features is None:
        raise ValueError("Debés especificar las columnas de características y la columna target.")

    df_features = df[col_features].apply(pd.to_numeric, errors='coerce')
    df_target = df[[col_target]] if isinstance(col_target, str) else df[col_target]

    df_clean = pd.concat([df_features, df_target], axis=1).dropna()

    X = df_clean[col_features].values
    y = df_clean[col_target].values.ravel()

    # Binarizar solo si se solicita
    if binarizar:
        if clase_minoria is None:
            raise ValueError("Debe indicarse la clase minoritaria si se va a binarizar.")
        y = np.where(y == clase_minoria, 1, 0)

    return X, y, None  # None en lugar de 'clases' explícitas


def graficar_distribucion_clases(y, nombre_dataset, clases_labels=None, guardar_en=None):
    conteo = Counter(y)
    clases = list(conteo.keys())
    cantidades = list(conteo.values())
    
    if clases_labels:
        clases = [clases_labels[c] if c in clases_labels else c for c in clases]
    
    plt.figure(figsize=(8, 5))
    plt.bar(clases, cantidades, color='skyblue')
    plt.xlabel("Clases")
    plt.ylabel("Cantidad de instancias")
    plt.title(f"Distribución de clases - {nombre_dataset}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if guardar_en:
        plt.savefig(guardar_en, dpi=300)
    plt.close()
