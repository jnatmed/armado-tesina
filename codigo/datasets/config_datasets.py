config_datasets = {
    # SHUTTLE (CSV con header correcto)
    "shuttle": {
        "path": "../datasets/statlog+shuttle/shuttle.csv",
        "dataset_name": "shuttle",
        "clase_minoria": 6,                         
        "clases_minor": [2, 6, 7],
        "col_features": [
                "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"
        ],
        "col_target": "target",
        "sep": ",",
        "header": 0,
        "binarizar": False,
        "tipo": "tabular"
    },

    # WDBC (sin header en archivo original .data)
    "wdbc": {
        "path": "../datasets/breast+cancer+wisconsin+original/wdbc.data",
        "dataset_name": "wdbc",
        "clase_minoria": "M",
        "col_target": "diagnosis",
        "col_features": [
            "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
            "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
            "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
            "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
            "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
            "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
        ],
        "sep": ",",
        "header": None,          # ← el esquema pondrá los nombres
        "binarizar": False,
        "tipo": "tabular"
    },

    # GLASS (sin header)
    "glass": {
        "path": "../datasets/glass+identification/glass.data",
        "dataset_name": "glass",
        "clase_minoria": 6,
        "col_target": "Type",
        "col_features": ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"],
        "sep": ",",
        "header": None,
        "binarizar": False,
        "tipo": "tabular"
    },

    # HEART (Cleveland, sin header)
    "heart": {
        "path": "../datasets/heart+disease/processed.cleveland.data",
        "dataset_name": "heart",
        "clase_minoria": 4,
        "col_target": "target",
        "col_features": [
            "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
            "oldpeak","slope","ca","thal"
        ],
        "sep": ",",
        "header": None,
        "binarizar": False,
        "tipo": "tabular"
    },

    # IRIS (sin header). Dataset balanceado → no definas minoritaria para EDA
    "iris": {
        "path": "../datasets/iris/iris.data",
        "dataset_name": "iris",
        "clase_minoria": None,   # ✅ quitar para evitar “posible error” artificial
        "col_target": "class",
        "col_features": ["sepal_length","sepal_width","petal_length","petal_width"],
        "sep": ",",
        "header": None,
        "binarizar": False,
        "tipo": "tabular"
    },

    # ECOLI (delimitado por espacios, sin header)
    "ecoli": {
        "path": "../datasets/ecoli/ecoli.data",
        "dataset_name": "ecoli",
        "clase_minoria": "imL",  # ✅ corregido (era "imS")
        "col_target": "class",
        "col_features": ["mcg","gvh","lip","chg","aac","alm1","alm2"],
        "sep": r"\s+",
        "header": None,
        "binarizar": False,
        "tipo": "tabular"
    },
}
