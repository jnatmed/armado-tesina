from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationCleaner:
    """
    Utilidad para limpiar outliers con IsolationForest.
    Soporta modo global o por clase.
    """

    @staticmethod
    def limpiarOutliers(X, y, modo="por_clase", contamination="auto",
                        n_estimators=200, max_samples="auto",
                        random_state=42, bootstrap=False, verbose=True):
        """
        Elimina outliers de X,y usando IsolationForest.
        Si modo='por_clase', entrena un IF separado para cada clase.

        Retorna:
            X_clean, y_clean
        """
        X = np.asarray(X)
        y = np.asarray(y)
        original_len = len(y)

        def _fit_predict_if(Xsub):
            iforest = IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=contamination,
                random_state=random_state,
                bootstrap=bootstrap,
                n_jobs=1
            )
            return iforest.fit_predict(Xsub)

        if modo == "por_clase":
            keep_mask = np.zeros_like(y, dtype=bool)
            removed_total = 0
            clases = np.unique(y)

            for c in clases:
                idx = np.where(y == c)[0]
                if len(idx) < 10:
                    keep_mask[idx] = True
                    continue
                preds = _fit_predict_if(X[idx])
                keep_mask[idx] = (preds == 1)
                removed_total += int(np.sum(preds == -1))

            X_clean = X[keep_mask]
            y_clean = y[keep_mask]
            if verbose:
                print(f"🧹 IsolationForest (por_clase): removidos {removed_total} outliers "
                      f"({original_len - len(y_clean)} total); "
                      f"train queda en {len(y_clean)} filas.")
        else:
            preds = _fit_predict_if(X)
            keep_mask = (preds == 1)
            X_clean = X[keep_mask]
            y_clean = y[keep_mask]
            removed_total = int(np.sum(~keep_mask))
            if verbose:
                print(f"🧹 IsolationForest (global): removidos {removed_total} outliers; "
                      f"train queda en {len(y_clean)} filas.")

        return X_clean, y_clean
