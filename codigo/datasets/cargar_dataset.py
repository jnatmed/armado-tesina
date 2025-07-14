import pandas as pd
import numpy as np

def cargar_dataset(path, clase_minoria=None, col_features=None, col_target=None, sep=' ', header=None):
    df = pd.read_csv(path, header=header, sep=sep)

    # Eliminar filas con valores faltantes explícitos
    df = df[~df.isin(['?']).any(axis=1)]

    if col_target is None or col_features is None:
        raise ValueError("Debés especificar las columnas de características y la columna target.")

    # Convertir a numérico solo las columnas de características
    df_features = df[col_features].apply(pd.to_numeric, errors='coerce')
    df_target = df[[col_target]] if isinstance(col_target, str) else df[col_target]

    # Concatenar para eliminar solo filas que tengan NaN en las features
    df_clean = pd.concat([df_features, df_target], axis=1).dropna()

    X = df_clean[col_features].values
    y = df_clean[col_target].values.ravel()

    # Binarizar la clase objetivo
    if clase_minoria is not None:
        y_bin = np.where(y == clase_minoria, 1, 0)
    else:
        raise ValueError("Debe indicarse la clase minoritaria")

    return X, y_bin
