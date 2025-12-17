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

<<<<<<< Updated upstream
    if col_target is None or col_features is None:
        raise ValueError("Debés especificar las columnas de características (col_features) y la columna target (col_target).")
=======
    # --- TABULAR NPZ (US CRIME, etc.) ---
    # --- TABULAR NPZ (US CRIME, etc.) ---
    if tipo == 'tabular_npz':
        data_npz = np.load(path)

        # Soportar dos formatos típicos:
        # 1) 'X' / 'y'
        # 2) 'data' / 'label'  ← el tuyo (US Crime)
        if 'X' in data_npz and 'y' in data_npz:
            X = data_npz['X']
            y = data_npz['y']
        elif 'data' in data_npz and 'label' in data_npz:
            X = data_npz['data']
            y = data_npz['label']
        else:
            raise KeyError(
                f"El archivo NPZ {path} debe contener ('X','y') o ('data','label'). "
                f"Claves encontradas: {data_npz.files}"
            )

        # Asegurar vector 1D
        y = np.asarray(y).ravel()
        X = np.asarray(X)

        # Definir nombres de columnas si no vienen dados
        n_features = X.shape[1]
        if col_features is None:
            col_features = [f"feat_{i}" for i in range(n_features)]

        # Construir DataFrame de features
        df_features = pd.DataFrame(X, columns=col_features)
        df_features = df_features.astype('float32')

        # Chequeo de NaN/inf en X
        matriz = df_features.to_numpy()
        if not np.isfinite(matriz).all():
            raise ValueError("❌ X (NPZ) contiene NaN o infinitos luego del preprocesamiento.")

        # Binarización (si corresponde)
        if binarizar:
            if clase_minoria is None:
                raise ValueError("Debe indicarse clase_minoria si se va a binarizar.")
            y = np.where(y == clase_minoria, 1, 0).astype(int)
            clases = np.array([0, 1])
        else:
            clases = np.unique(y)

        return df_features, y, clases

    # ───────────────────────── TABULAR (CSV/ESPACIOS) ─────────────────────────

    # --- Validaciones mínimas para tabular clásico ---
    if col_target is None:
        raise ValueError("Debés especificar col_target para tipo='tabular'.")

    # Si no se pasan features, usar todas excepto el target
    if col_features is None:
        # Ojo: esto debe ejecutarse DESPUÉS de cargar el df
        pass

>>>>>>> Stashed changes

    # 1) Leer declarando NA explícitos (evita que 'ca'/'thal' queden como object en Cleveland Heart)
    df = pd.read_csv(path, header=header, sep=sep, na_values=list(na_values), engine='python')

<<<<<<< Updated upstream
    # 2) Seleccionar features y target
    #    (funciona con nombres de columna o índices enteros)
=======
    # --- Lectura robusta ---
    if usar_names is not None:
        df = pd.read_csv(
            path,
            header=None,
            names=usar_names,
            sep=sep,
            na_values=list(na_values),
            engine='python',
            skipinitialspace=True
        )
    else:
        df = pd.read_csv(
            path,
            header=header,
            sep=sep,
            na_values=list(na_values),
            engine='python',
            skipinitialspace=True
        )

    if col_features is None:
        col_features = [c for c in df.columns if c != col_target]


    # --- Anti-encabezado duplicado (si se pasó names y header=None) ---
    if usar_names is not None and header is None and len(df) > 0:
        fila0 = df.iloc[0].astype(str).str.strip().tolist()
        esquema = [str(c).strip() for c in usar_names]
        coincidencias = 0
        i = 0
        limite = min(len(fila0), len(esquema))
        while i < limite:
            if fila0[i] == esquema[i]:
                coincidencias += 1
            i += 1
        umbral = max(3, int(0.6 * limite))
        if coincidencias >= umbral:
            df = df.iloc[1:].reset_index(drop=True)

    # --- Mapeo automático por dataset_name si no se usó names y coincide cantidad ---
    if header is None and usar_names is None and dataset_name in ESQUEMAS_CONOCIDOS:
        esquema = ESQUEMAS_CONOCIDOS[dataset_name]
        if len(esquema) == df.shape[1]:
            df.columns = esquema

    # --- Validación de columnas requeridas (evita KeyError crípticos) ---
    columnas_requeridas = []
    for c in col_features:
        columnas_requeridas.append(c)
    if isinstance(col_target, str):
        columnas_requeridas.append(col_target)
    else:
        for c in col_target:
            columnas_requeridas.append(c)

    cols_faltantes = []
    for c in columnas_requeridas:
        if c not in df.columns:
            cols_faltantes.append(c)

    if len(cols_faltantes) > 0:
        primeras = df.columns.tolist()
        primeras = primeras[:min(20, len(primeras))]
        raise KeyError(
            f"Columnas ausentes: {cols_faltantes}. "
            f"Leídas={len(df.columns)} → {primeras}..."
        )


    print("[DEBUG] path:", path)
    print("[DEBUG] sep:", sep, "header:", header)
    print("[DEBUG] columnas:", df.columns.tolist())
    print("[DEBUG] dtypes antes to_numeric:\n", df[col_features].dtypes)
    print("[DEBUG] head features:\n", df[col_features].head(3))

    # --- Selección de features y target ---
>>>>>>> Stashed changes
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
