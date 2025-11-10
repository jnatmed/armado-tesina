ESQUEMAS_CONOCIDOS = {
    "iris": [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
    ],
    "wdbc": [  # Breast Cancer Wisconsin (Diagnostic)
        # 31 columnas: id, diagnosis + 30 features
        "id", "diagnosis",
        "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
        "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
        "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
        "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
        "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
    ],
    "glass": [
        "RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type"
    ],
    "ecoli": [  # Ecoli UCI (7 attrs + class)
        "mcg","gvh","lip","chg","aac","alm1","alm2","class"
    ],
    "heart": [  # Cleveland Heart (14 attrs + class)
        "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
        "oldpeak","slope","ca","thal","target"
    ],
    "shuttle": [  # NASA Shuttle (9 attrs + class)
        "A1","A2","A3","A4","A5","A6","A7","A8","A9","Class"
    ],
}
