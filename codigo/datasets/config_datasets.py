config_datasets = {
    # Dataset ECOLI - Multiclase (8 clases)
    "ecoli": {
        "path": "../datasets/ecoli/ecoli.data",
        "clase_minoria": "imL",  # Coincide con la clase minoritaria real
        "col_features": list(range(1, 8)),  # Columnas 1 a 7 como features
        "col_target": 8,  # Columna 8 es el target
        "sep": r'\s+',  # Separador por espacios
        "header": None
    },
    # Dataset WDBC (Breast Cancer Wisconsin) - Binario
    "wdbc": {
        "path": "../datasets/breast+cancer+wisconsin+original/wdbc.data",
        "clase_minoria": "M",  # ✅ Coincide con la clase minoritaria real
        "col_features": list(range(2, 32)),  # Columnas 2 a 31 (30 features)
        "col_target": 1,  # Columna 1 es el target (B/M)
        "sep": ",",
        "header": None
    },
    # Dataset GLASS - Multiclase (6 clases)
    "glass": {
        "path": "../datasets/glass+identification/glass.data",
        "clase_minoria": 6,  # ✅ Coincide con la clase minoritaria real
        "col_features": list(range(1, 10)),  # Columnas 1 a 9 como features
        "col_target": 10,  # Columna 10 es el target
        "sep": ",",
        "header": None
    },
    # Dataset HEART - Multiclase (5 clases)
    "heart": {
        "path": "../datasets/heart+disease/processed.cleveland.data",
        "clase_minoria": 4,  # clase minoritaria real        
        "col_features": list(range(0, 13)),
        "col_target": 13,
        "sep": ",",
        "header": None
    },
    "eurosat": {
        "path": "../datasets/EuroSAT",
        "clase_minoria": 5,  # clase minoritaria real
        "tipo": "imagen",
        "size": (64, 64)
    }
    
}
