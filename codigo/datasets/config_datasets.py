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
    "ecoli": {
        "path": "../datasets/ecoli/ecoli.data",
        "clase_minoria": "imS",  # clase minoritaria real
        "col_features": list(range(1, 8)),  # columnas 1 a 7 como features
        "col_target": 8,  # columna 8 es el target
        "sep": "\s+",
        "header": None
    },
<<<<<<< Updated upstream
    # "eurosat": {
    #     "path": "../datasets/EuroSAT",
    #     "clase_minoria": 5,  # clase minoritaria real
    #     "tipo": "imagen",
    #     "size": (64, 64)
    # }
    
=======

    # ───────────────────────────── PREDICT_FAULTS ─────────────────────────────
    "predict_faults": {
        "path": "../datasets/predict_faults/predictive_maintenance.csv",
        "dataset_name": "predict_faults",

        "clase_minoria": "Random Failures",              # multiclase real
        "col_target": "Failure Type",

        # SOLO columnas numéricas útiles para el modelo
        "col_features": [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ],

        "sep": ",",
        "header": 0,
        "binarizar": False,
        "tipo": "tabular",

        "limpieza_outliers": {
            "activar": True,
            "estrategia": "progresiva",
            "niveles": {
                "nivel_1": {
                    "tipo": "rango_fisico",
                    "criterios": {},
                    "fail_safe_max_ratio_eliminados": 0.0
                },
                "nivel_2": {"tipo": "iqr_por_clase", "activar": True, "solo_marcar": True},
                "nivel_3": {"tipo": "isolation_forest", "activar": False}
            },
            "comentario": "No eliminar outliers: las colas representan fallas reales."
        },

        "transformacion": {
            "escalado": {"tipo": "robust", "aplicar": True}
        }
    },

    # ───────────────────────────── GEAR VIBRATION ─────────────────────────────
    "gear_vibration": {
        "path": "../datasets/gear_vibration/gear_vibration_operativo.csv",
        "dataset_name": "gear_vibration",

        "clase_minoria": None,
        "col_target": "label",

        "col_features": ["s1_media", "s1_std", "s1_rms", "s2_media", "s2_std", "s2_rms","s1_s2_corr", "speedSet", "load_value"],

        "sep": ",",
        "header": 0,
        "binarizar": False,
        "tipo": "tabular",

        "limpieza_outliers": {
            "activar": False,   # baseline primero
            "estrategia": "progresiva",
            "niveles": {
                "nivel_1": {"tipo": "rango_fisico", "criterios": {}, "fail_safe_max_ratio_eliminados": 0.0},
                "nivel_2": {"tipo": "iqr_por_clase", "activar": False, "solo_marcar": True},
                "nivel_3": {"tipo": "isolation_forest", "activar": False}
            },
            "comentario": "Primero baseline sin limpieza; luego ablation."
        },

        "transformacion": {
            "escalado": {"tipo": "standard", "aplicar": True}
        }
    },





>>>>>>> Stashed changes
}
