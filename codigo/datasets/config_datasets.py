config_datasets = {
    "ecoli": {
        "path": "../datasets/ecoli/ecoli.data",
        "clase_minoria": "imL",
        "col_features": list(range(1, 8)),
        "col_target": 8,
        "sep": r'\s+',
        "header": None
    },
    "wdbc": {
        "path": "../datasets/breast+cancer+wisconsin+original/wdbc.data",
        "clase_minoria": "M",  
        "col_features": list(range(2, 32)),  # columnas 2 a 31 (30 features)
        "col_target": 1,  # la columna con B/M
        "sep": ",",
        "header": None
    },
    "glass": {
        "path": "../datasets/glass+identification/glass.data",
        "clase_minoria": 6,  # Ejemplo: clase 6 es muy poco frecuente (puede ajustarse)
        "col_features": list(range(1, 10)),  # del 1 al 9
        "col_target": 10,
        "sep": ",",
        "header": None
    },
    "heart": {
        "path": "../datasets/heart+disease/processed.cleveland.data",
        "clase_minoria": 1,  # 1 = presencia de enfermedad (si convertís y > 0 → 1)
        "col_features": list(range(0, 13)),  # 13 columnas
        "col_target": 13,
        "sep": ",",
        "header": None
    },
    # "diabetes": {
    #     "path": "../datasets/diabetes/diabetes.csv",
    #     "clase_minoria": 1,
    #     "col_features": list(range(0, 8)),
    #     "col_target": 8,
    #     "sep": ',',
    #     "header": 0
    # },
    
}
