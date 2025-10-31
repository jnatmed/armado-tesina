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
                n_estimators=n_estimators, # Es la cantidad de árboles del bosque
                max_samples=max_samples, # Es la cantidad de muestras para entrenar cada árbol
                contamination=contamination, # Es la proporción de outliers esperados
                random_state=random_state, 
                bootstrap=bootstrap, # si se hace muestreo con reemplazo
                n_jobs=1
            )
            # genera un array con etiquetas 1 (inlier) o -1 (outlier)
            return iforest.fit_predict(Xsub)

        # limpieza de outliers por clase o global
        if modo == "por_clase":
            # trabaja como máscara booleana, True = keep, False = remove
            # parecido a como trabaja pcsmote en fit_resample_multiclass
            # donde se arma una máscara por clase, el resto a la clase
            # actual se pone en 0, y se realiza la aumentacion sobre esta
            # mascara.
            keep_mask = np.zeros_like(y, dtype=bool)
            # contador de outliers removidos
            removed_total = 0
            # iterar por clases
            clases = np.unique(y)
            # por cada clase iterar y entrenar un IF
            for c in clases:
                idx = np.where(y == c)[0]
                # Si la clase tiene menos de 10 muestras, no se aplica IF
                # porque no hay suficientes datos para entrenar
                if len(idx) < 10:
                    # todas las muestras se mantienen
                    keep_mask[idx] = True
                    continue
                """
                 Caso contrario se entrena el IF sobre las muestras
                 de la clase actual y se predicen outliers
                """
                preds = _fit_predict_if(X[idx])
                """
                Pondra true en las muestras normales (preds==1)
                y false en las outliers (preds==-1)
                """
                keep_mask[idx] = (preds == 1)
                # contar outliers removidos
                removed_total += int(np.sum(preds == -1))

            X_clean = X[keep_mask]
            y_clean = y[keep_mask]
            if verbose:
                print(f"🧹 IsolationForest (por_clase): removidos {removed_total} outliers "
                      f"({original_len - len(y_clean)} total); "
                      f"train queda en {len(y_clean)} filas.")
        else:
            """modo global, se entrena un IF sobre todo el dataset"""
            preds = _fit_predict_if(X)
            # aplicara true sobre las muestras normales (preds==1)
            keep_mask = (preds == 1)
            X_clean = X[keep_mask]
            y_clean = y[keep_mask]
            # contar outliers removidos
            removed_total = int(np.sum(~keep_mask))
            if verbose:
                print(f"🧹 IsolationForest (global): removidos {removed_total} outliers; "
                      f"train queda en {len(y_clean)} filas.")

        return X_clean, y_clean
