import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from cargar_eurosat import cargar_dataset_eurosat  

def cargar_dataset(path, clase_minoria=None, col_features=None, col_target=None,
                   sep=' ', header=None, binarizar=True, tipo='tabular',
                   impute='median',                 # 'median' o 'drop'
                   na_values=('?', 'NA', 'None')):  # tokens a tratar como NaN
    """
    Carga datasets de tipo 'tabular' o 'imagen'.
    - Tabular: convierte features a numérico, maneja NA (imputación o drop) y retorna (X, y, None).
    - Imagen: delega a cargar_dataset_eurosat(path).
    Si binarizar=True, transforma y en {0,1} usando clase_minoria; si False, deja y multiclase.
    """
    if tipo == 'imagen':
        X, y, clases = cargar_dataset_eurosat(path)
        return X, y, clases

    if col_target is None or col_features is None:
        raise ValueError("Debés especificar las columnas de características (col_features) y la columna target (col_target).")

    # 1) Leer declarando NA explícitos (evita que 'ca'/'thal' queden como object en Cleveland Heart)
    df = pd.read_csv(path, header=header, sep=sep, na_values=list(na_values), engine='python')

    # 2) Seleccionar features y target
    #    (funciona con nombres de columna o índices enteros)
    df_features = df[col_features].apply(pd.to_numeric, errors='coerce')
    df_target = df[[col_target]] if isinstance(col_target, str) else df[col_target]

    # 3) Imputación o drop de NA
    if impute == 'drop':
        mask_valid = df_features.notna().all(axis=1) & df_target.notna().all(axis=1)
        df_features = df_features.loc[mask_valid]
        df_target = df_target.loc[mask_valid]
    elif impute == 'median':
        med = df_features.median(numeric_only=True)
        df_features = df_features.fillna(med)
        # target no debería tener NA; si los hay, se dropean esas filas
        df_target = df_target.dropna(axis=0)
        df_features = df_features.loc[df_target.index]
    else:
        raise ValueError("impute debe ser 'median' o 'drop'.")

    # 4) Advertencia si, tras convertir, quedó alguna columna no numérica (no debería)
    tipos_invalidos = df_features.select_dtypes(include=['object']).columns
    if len(tipos_invalidos) > 0:
        print(f"⚠️ Advertencia: columnas no numéricas detectadas tras conversión: {list(tipos_invalidos)}")

    # 5) Arrays finales
    X = df_features.to_numpy(dtype=np.float32)
    y = df_target.values.ravel()

    # 6) Validaciones
    if not np.isfinite(X).all():
        raise ValueError("❌ X contiene NaN o infinitos luego del preprocesamiento.")

    # 7) Binarización opcional (para runs binarios directos). Para multiclase, dejá binarizar=False.
    if binarizar:
        if clase_minoria is None:
            raise ValueError("Debe indicarse la clase_minoria si se va a binarizar.")
        y = np.where(y == clase_minoria, 1, 0).astype(int)

    return X, y, None  # 'clases' no se usa en tabular


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
