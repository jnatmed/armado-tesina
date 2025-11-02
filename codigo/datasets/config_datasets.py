config_datasets = {
    # Dataset SHUTTLE - Multiclase (7 clases, altamente desbalanceado)
    # Simula el estado de un sistema de control de un transbordador espacial.
    # La clase 1 representa el 78% de los datos, mientras que clases como la 6 y 7 tienen muy pocas muestras.
    # Ideal para evaluar la efectividad de técnicas de sobremuestreo en entornos multiclase severamente desbalanceados.
    "shuttle": {
        "path": "../datasets/statlog+shuttle/shuttle.csv",
        "clase_minoria": 7,  # Clase más pequeña (~15 instancias)
        "col_features": [f"feat_{i}" for i in range(9)],  # Columnas nombradas como strings
        "clases_minor": [2, 6, 7],  # ✅ Clases consideradas como minoritarias        
        "col_target": 'target',  # Columna "target"
        "sep": ",",
        "header": 0
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
    "iris": {
        "path": "../datasets/iris/iris.data",
        "clase_minoria": "Iris-virginica",      # suele tener menos instancias que setosa o versicolor
        "col_features": [0, 1, 2, 3],           # cuatro características numéricas
        "col_target": 4,                        # última columna con el nombre de la especie
        "sep": ",",
        "header": None,
        "tipo": "tabular"
    },    
    # "eurosat": {
    #     "path": "../datasets/EuroSAT",
    #     "clase_minoria": 5,  # clase minoritaria real
    #     "tipo": "imagen",
    #     "size": (64, 64)
    # }
    
}
