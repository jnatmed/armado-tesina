from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.preprocessing import LabelEncoder

class IsolationCleaner:
    """
    Limpieza de outliers *por clase* con IsolationForest usando UMBRALES DE SCORE.
    - Sin modo global.
    - Sin etiquetas -1/1 (no se usa fit_predict): solo decision_function >= umbral.
    - Umbral por dataset y clase vía 'umbrales' + 'dataset_name'.
    - Opcional: normalizar scores a [-1,1] ANTES de umbralizar (consistente con tu estudio).
    """

    # ---------- Helpers ----------
    @staticmethod
    def construir_umbrales_para_cleaner(umbrales_por_dataset, nombre_dataset, y, encoder: LabelEncoder | None):
        """
        Devuelve: { nombre_dataset: { clase_codificada(str): umbral(float), ... } }
        - Si las clases ya son numéricas, solo castea las claves a str.
        - Si eran strings (p.ej. 'B','M'), las traduce al ID del LabelEncoder.
        """
        ds = umbrales_por_dataset.get(nombre_dataset, {})
        if not ds:
            return None

        if encoder is None:
            return {nombre_dataset: {str(k): float(v) for k, v in ds.items()}}

        umbrales_map = {}
        for k, v in ds.items():
            try:
                cls_id = int(encoder.transform([k])[0])
                umbrales_map[str(cls_id)] = float(v)
            except ValueError:
                continue
        return {nombre_dataset: umbrales_map} if umbrales_map else None

    @staticmethod
    def _resolver_umbral_clase(dataset_name, clase, umbrales, fallback):
        """
        Prioridad:
          1) umbrales[dataset_name][clase]
          2) fallback (umbral_score o 0.0)
        """
        if not umbrales or not dataset_name:
            return fallback
        d = umbrales.get(dataset_name)
        if isinstance(d, dict):
            key = str(clase)
            if key in d:
                return float(d[key])
        return fallback

    # ---------- Limpieza por clase ----------
    @staticmethod
    def limpiarOutliers(
        X, y,
        # compat: aceptamos 'modo' pero solo por_clase; se ignora
        modo: str | None = None,
        contamination="auto",
        n_estimators=200,
        max_samples="auto",
        random_state=42,
        bootstrap=False,

        # compat: aceptamos estos kwargs (aunque siempre trabajamos con score)
        usar_score: bool = True,
        umbral_score=None,
        normalizar_scores: bool = False,

        # umbrales por dataset/clase
        umbrales=None,
        dataset_name=None,

        devolver_info=False,
        verbose=True,
    ):
        if modo is not None and modo != "por_clase":
            raise ValueError("IsolationCleaner: solo se soporta 'por_clase'.")

        X = np.asarray(X); y = np.asarray(y)
        original_len = len(y)

        def _fit_scores(Xsub):
            iforest = IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=contamination,
                random_state=random_state,
                bootstrap=bootstrap,
                n_jobs=1
            )
            iforest.fit(Xsub)
            s = iforest.decision_function(Xsub)  # >0 inlier; <0 outlier
            if normalizar_scores:
                s_min, s_max = np.min(s), np.max(s)
                s = (2.0 * (s - s_min) / (s_max - s_min) - 1.0) if s_max > s_min else np.zeros_like(s)
            return s

        # ---- único camino: por clase con score ----
        keep_mask = np.zeros_like(y, dtype=bool)
        removed_total = 0
        clases = np.unique(y)
        scores_full = np.full(shape=(len(y),), fill_value=np.nan, dtype=float)
        umbral_usado_por_clase = {}
        fallback = 0.0 if umbral_score is None else float(umbral_score)

        for c in clases:
            idx = np.where(y == c)[0]
            if len(idx) < 10:
                keep_mask[idx] = True
                umbral_usado_por_clase[c] = None
                continue

            s_local = _fit_scores(X[idx])
            thr = IsolationCleaner._resolver_umbral_clase(dataset_name, c, umbrales, fallback)
            keep_local = (s_local >= thr)

            scores_full[idx] = s_local
            umbral_usado_por_clase[c] = thr
            keep_mask[idx] = keep_local
            removed_total += int(np.sum(~keep_local))

        X_clean = X[keep_mask]; y_clean = y[keep_mask]

        if verbose:
            print(f"🧹 IF (por_clase): removidos {removed_total}; quedan {len(y_clean)} filas.")

        if devolver_info:
            info = {
                "scores": scores_full[keep_mask],
                "idx_keep": np.where(keep_mask)[0],
                "idx_removed": np.where(~keep_mask)[0],   # <- añadimos eliminadas
                "umbrales_por_clase": umbral_usado_por_clase,
                "removed_total": removed_total,
            }
            return X_clean, y_clean, info
        return X_clean, y_clean