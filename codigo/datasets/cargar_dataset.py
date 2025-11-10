import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from cargar_eurosat import cargar_dataset_eurosat  

from esquemas_conocidos import ESQUEMAS_CONOCIDOS

def cargar_dataset(path, clase_minoria=None, col_features=None, col_target=None,
                   sep=' ', header=None, binarizar=True, tipo='tabular',
                   impute='median', na_values=('?', 'NA', 'None'),
                   dataset_name=None, names=None):
    """
    ... (docstring igual) ...
    - dataset_name: clave para aplicar ESQUEMAS_CONOCIDOS si header=None.
    - names: lista explícita de nombres para read_csv (override).
    """
    if tipo == 'imagen':
        X, y, clases = cargar_dataset_eurosat(path)
        return X, y, clases

    if col_target is None or col_features is None:
        raise ValueError("Debés especificar col_features y col_target.")

    # ---- Selección de 'names' para read_csv ----
    usar_names = None
    if names is not None:
        usar_names = list(names)
    elif header is None and dataset_name is not None:
        if dataset_name in ESQUEMAS_CONOCIDOS:
            usar_names = ESQUEMAS_CONOCIDOS[dataset_name]

    # Si pasamos 'names', debe ser coherente con columnas del archivo
    if usar_names is not None:
        df = pd.read_csv(path, header=None, names=usar_names, sep=sep,
                         na_values=list(na_values), engine='python')
    else:
        df = pd.read_csv(path, header=header, sep=sep,
                         na_values=list(na_values), engine='python')

    # ---- (OPCIONAL) Autodetección si header=None y cantidad coincide ----
    if header is None and usar_names is None and dataset_name in ESQUEMAS_CONOCIDOS:
        esquema = ESQUEMAS_CONOCIDOS[dataset_name]
        if len(esquema) == df.shape[1]:
            df.columns = esquema  # mapeo automático

    # Selección features y target (por nombre o índice)
    df_features = df[col_features].apply(pd.to_numeric, errors='coerce')
    df_target = df[[col_target]] if isinstance(col_target, str) else df[col_target]

    # Imputación / drop
    if impute == 'drop':
        mask_valid = df_features.notna().all(axis=1) & df_target.notna().all(axis=1)
        df_features = df_features.loc[mask_valid]
        df_target = df_target.loc[mask_valid]
    elif impute == 'median':
        med = df_features.median(numeric_only=True)
        df_features = df_features.fillna(med)
        df_target = df_target.dropna(axis=0)
        df_features = df_features.loc[df_target.index]
    else:
        raise ValueError("impute debe ser 'median' o 'drop'.")

    df_features = df_features.astype('float32')
    y = df_target.values.ravel()

    if not np.isfinite(df_features.to_numpy()).all():
        raise ValueError("❌ X contiene NaN o infinitos luego del preprocesamiento.")

    if binarizar:
        if clase_minoria is None:
            raise ValueError("Debe indicarse clase_minoria si se va a binarizar.")
        y = np.where(y == clase_minoria, 1, 0).astype(int)
        clases = np.array([0, 1])
    else:
        clases = pd.Series(y).unique()

    if isinstance(col_features[0], str):
        df_features.columns = col_features

    # 👉 devolvemos DataFrame con NOMBRES reales
    return df_features, y, clases




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
