from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from scipy.stats import entropy
from collections import Counter
import numpy as np
import pandas as pd
import json


class PCSMOTE:
    """
    PC-SMOTE con:
      - Topes de crecimiento (global y por clase).
      - Logs por clase enriquecidos (umbrales, riesgo, densidad, filtradas).
      - Metadatos de experimento exportables a JSON.
    """

    def __init__(self, k_neighbors=5, random_state=None,
                 radio_densidad=1.0, percentil_dist=75,
                 percentil_entropia=None, percentil_densidad=None,
                 criterio_pureza='entropia', modo_espacial='2d',
                 factor_equilibrio=0.8, verbose=True,
                 max_total_multiplier=None,       # límite |dataset_final| ≤ mult * |dataset_original|
                 max_sinteticas_por_clase=None):  # límite duro por clase
        # Hiperparámetros
        self.k = int(k_neighbors)
        self._seed_init = random_state
        self.random_state = check_random_state(random_state)
        self.radio_densidad = float(radio_densidad)
        self.percentil_dist = float(percentil_dist)
        self.percentil_entropia = None if percentil_entropia is None else float(percentil_entropia)
        self.percentil_densidad = None if percentil_densidad is None else float(percentil_densidad)
        self.criterio_pureza = str(criterio_pureza)
        self.modo_espacial = str(modo_espacial).lower()  # '2d' o '3d'
        self.factor_equilibrio = float(factor_equilibrio)
        self.verbose = bool(verbose)

        # Topes
        self.max_total_multiplier = None if max_total_multiplier is None else float(max_total_multiplier)
        self.max_sinteticas_por_clase = None if max_sinteticas_por_clase is None else int(max_sinteticas_por_clase)

        # Logging
        self.logs_por_clase = []
        self.meta_experimento = {}
        self._meta = {}  # métricas de la última llamada a fit_resample

        # Nombre del dataset (opcional; si no, queda 'unknown')
        self.nombre_dataset = getattr(self, "nombre_dataset", "unknown")

    # --------------------- Utilidades ---------------------

    def reset_logs(self):
        self.logs_por_clase = []
        self.meta_experimento = {}
        self._meta = {}

    def _loggable_random_state(self):
        if isinstance(self._seed_init, (int, np.integer)):
            return int(self._seed_init)
        if self._seed_init is None:
            return None
        return str(self._seed_init)

    def exportar_log_csv(self, path_salida):
        if not self.logs_por_clase:
            print("⚠️ No hay log de sobremuestreo para exportar.")
            return
        df = pd.DataFrame(self.logs_por_clase)
        df.to_csv(path_salida, index=False)
        print(f"📁 Log de sobremuestreo guardado en: {path_salida}")

    def exportar_log_json_meta(self, path_json):
        if not self.meta_experimento:
            print("⚠️ No hay metadatos de experimento para exportar.")
            return
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(self.meta_experimento, f, ensure_ascii=False, indent=2)
        print(f"📁 Metadatos de experimento guardados en: {path_json}")

    # --------------------- Cálculos auxiliares ---------------------

    def calcular_densidad_interseccion(self, X_min, vecinos, radio):
        """
        Densidad por intersección de esferas de radio 'radio'.
        En modo 3D utiliza las primeras 3 columnas como coordenadas espaciales.
        """
        densidades = []
        for i, xi in enumerate(X_min):
            intersecciones = 0
            for j in vecinos[i]:
                xj = X_min[j]
                if self.modo_espacial == '3d':
                    distancia = np.linalg.norm(xi[:3] - xj[:3])
                else:
                    distancia = np.linalg.norm(xi - xj)
                if distancia <= 2.0 * radio:
                    intersecciones += 1
            densidades.append(intersecciones / max(1, len(vecinos[i])))
        return np.array(densidades, dtype=float)

    def calcular_entropia(self, vecinos, y):
        entropias = []
        for idxs in vecinos:
            clases, counts = np.unique(y[idxs], return_counts=True)
            p = counts / counts.sum()
            entropias.append(float(entropy(p, base=2)))
        return np.array(entropias, dtype=float)

    # --------------------- Núcleo binario ---------------------

    def fit_resample(self, X, y, max_sinteticas=None):
        """
        Resample binario: y ∈ {0,1} (1 = minoritaria). Devuelve (X_resampled, y_resampled).
        Guarda métricas del run en self._meta (para ser reutilizadas por el multiclass).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Inicializar meta para este run
        self._meta = {
            "umbral_densidad": None,
            "umbral_entropia": None,
            "riesgo_medio": None,
            "riesgo_std": None,
            "densidad_media": None,
            "vecinos_validos_promedio": None,
            "n_candidatas": None,
            "n_filtradas": None
        }

        # Separación por clase binaria
        X_min = X[y == 1]
        X_maj = X[y == 0]

        if len(X_min) < self.k + 1:
            # No alcanza para vecinos: devolver copia
            self._meta.update({
                "n_candidatas": int(len(X_min)),
                "n_filtradas": 0
            })
            return X.copy(), y.copy()

        # Vecinos globales (para riesgo) y vecinos dentro minoritaria (para densidad)
        nn_all = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        vecinos_all = nn_all.kneighbors(X_min, return_distance=False)[:, 1:]

        riesgo = np.array([np.sum(y[idxs] == 0) / self.k for idxs in vecinos_all], dtype=float)

        nn_min = NearestNeighbors(n_neighbors=self.k + 1).fit(X_min)
        vecinos_min = nn_min.kneighbors(X_min, return_distance=False)[:, 1:]
        densidades = self.calcular_densidad_interseccion(X_min, vecinos_min, self.radio_densidad)

        # Criterio pureza
        if self.criterio_pureza == 'entropia':
            entropias = self.calcular_entropia(vecinos_all, y)
            if self.percentil_entropia is not None:
                umb_ent = float(np.percentile(entropias, self.percentil_entropia))
            else:
                umb_ent = None
            pureza_mask = entropias <= (umb_ent if umb_ent is not None else 1.0)
            self._meta["umbral_entropia"] = umb_ent
        elif self.criterio_pureza == 'proporcion':
            proporciones_min = np.array([np.sum(y[idxs] == 1) / self.k for idxs in vecinos_all], dtype=float)
            pureza_mask = (proporciones_min >= 0.4) & (proporciones_min <= 0.6)
        else:
            raise ValueError(f"Criterio de pureza no reconocido: {self.criterio_pureza}")

        # Criterio densidad
        if self.percentil_densidad is not None:
            umb_den = float(np.percentile(densidades, self.percentil_densidad))
            densidad_mask = densidades >= umb_den
            self._meta["umbral_densidad"] = umb_den
        else:
            umb_den = None
            densidad_mask = densidades > 0.0

        # Combinación de filtros
        comb = pureza_mask & densidad_mask
        X_min_filtrado = X_min[comb]

        # Métricas agregadas (antes de generar)
        self._meta.update({
            "n_candidatas": int(len(X_min)),
            "n_filtradas": int(np.sum(comb)),
            "riesgo_medio": float(np.mean(riesgo[comb])) if np.any(comb) else None,
            "riesgo_std": float(np.std(riesgo[comb])) if np.any(comb) else None,
            "densidad_media": float(np.mean(densidades)) if densidades.size else None
        })

        # Vecinos válidos promedio (por percentil_dist) — diagnóstico
        if np.any(comb):
            counts = []
            for i, idxs_vecinos in enumerate(vecinos_all[comb]):
                xi = X_min_filtrado[i]
                if self.modo_espacial == '3d':
                    distancias = np.linalg.norm(X[idxs_vecinos][:, :3] - xi[:3], axis=1)
                else:
                    distancias = np.linalg.norm(X[idxs_vecinos] - xi, axis=1)
                thr = np.percentile(distancias, self.percentil_dist)
                counts.append(int(np.sum(distancias <= thr)))
            self._meta["vecinos_validos_promedio"] = float(np.mean(counts)) if counts else None
        else:
            self._meta["vecinos_validos_promedio"] = None

        if len(X_min_filtrado) < self.k + 1:
            return X.copy(), y.copy()

        # Cantidad de sintéticas
        n_sint = max_sinteticas if max_sinteticas is not None else (len(X_maj) - len(X_min))
        n_sint = int(max(0, n_sint))
        if n_sint == 0:
            return X.copy(), y.copy()

        # Generación de sintéticas
        muestras_sinteticas = []
        for _ in range(n_sint):
            idx = self.random_state.randint(len(X_min_filtrado))
            xi = X_min_filtrado[idx]
            r_i = riesgo[comb][idx]
            idxs_vecinos = vecinos_all[comb][idx]

            if self.modo_espacial == '3d':
                distancias = np.linalg.norm(X[idxs_vecinos][:, :3] - xi[:3], axis=1)
            else:
                distancias = np.linalg.norm(X[idxs_vecinos] - xi, axis=1)

            thr = np.percentile(distancias, self.percentil_dist)
            vecinos_validos = idxs_vecinos[distancias <= thr]
            if len(vecinos_validos) == 0:
                continue

            z_idx = self.random_state.choice(vecinos_validos)
            xz = X[z_idx]

            # Delta según riesgo local
            if 0.4 <= r_i < 0.5:
                delta = self.random_state.uniform(0.6, 0.8)
            elif 0.5 <= r_i <= 0.6:
                delta = self.random_state.uniform(0.3, 0.5)
            else:
                delta = self.random_state.uniform(0.4, 0.6)

            muestras_sinteticas.append(xi + delta * (xz - xi))

        if not muestras_sinteticas:
            return X.copy(), y.copy()

        X_sint = np.asarray(muestras_sinteticas, dtype=float)
        y_sint = np.ones(len(X_sint), dtype=int)

        X_resampled = np.vstack([X, X_sint])
        y_resampled = np.hstack([y, y_sint])
        return X_resampled, y_resampled

    # --------------------- Multiclase ---------------------

    def fit_resample_multiclass(self, X, y):
        """
        Extiende a multiclase sobremuestreando cada clase contra el máximo,
        con factor_equilibrio. Aplica topes si están configurados.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        clases = np.unique(y)
        X_res = X.copy()
        y_res = y.copy()

        total_original = len(y)
        conteo_original = Counter(y)
        max_count = max(conteo_original.values())

        # Metadatos globales del experimento
        self.meta_experimento = {
            "dataset": self.nombre_dataset,
            "k_neighbors": self.k,
            "radio_densidad": self.radio_densidad,
            "percentil_dist": self.percentil_dist,
            "percentil_entropia": self.percentil_entropia,
            "percentil_densidad": self.percentil_densidad,
            "criterio_pureza": self.criterio_pureza,
            "modo_espacial": self.modo_espacial,
            "factor_equilibrio": self.factor_equilibrio,
            "max_total_multiplier": self.max_total_multiplier,
            "max_sinteticas_por_clase": self.max_sinteticas_por_clase,
            "random_state": self._loggable_random_state(),
            "timestamp": pd.Timestamp.now().isoformat()
        }

        for clase in clases:
            y_bin = (y == clase).astype(int)
            actual = int(np.sum(y_bin))
            objetivo = int(max_count * self.factor_equilibrio)
            estado = "sobremuestreada" if actual < objetivo else "no se sobremuestrea"

            faltante_solicitado = max(0, objetivo - actual)
            faltante = max(0, objetivo - actual)

            # --- flags de topes (para diagnóstico)
            tope_por_clase_aplicado = False
            tope_global_aplicado = False

            # Tope por clase
            if self.max_sinteticas_por_clase is not None:
                nuevo_faltante = min(faltante, self.max_sinteticas_por_clase)
                tope_por_clase_aplicado = (nuevo_faltante < faltante)
                faltante = nuevo_faltante

            # Tope global (tamaño final)
            if faltante > 0 and self.max_total_multiplier is not None:
                max_total = int(total_original * self.max_total_multiplier)
                margen_global = max_total - len(y_res)
                if margen_global <= 0:
                    if self.verbose:
                        print(f"⚠️ Tope global alcanzado ({max_total}). No se generan más sintéticas.")
                    tope_global_aplicado = True
                    faltante = 0
                else:
                    recorte = min(faltante, margen_global)
                    tope_global_aplicado = (recorte < faltante)
                    faltante = recorte                   

            nuevos = 0
            meta_clase = {
                "umbral_densidad": None,
                "umbral_entropia": None,
                "riesgo_medio": None,
                "riesgo_std": None,
                "densidad_media": None,
                "vecinos_validos_promedio": None,
                "n_candidatas": None,
                "n_filtradas": None
            }

            if faltante > 0:
                # Resample binario por clase (apoya logging interno en _meta)
                sampler_tmp = PCSMOTE(
                    k_neighbors=self.k,
                    random_state=self._seed_init,
                    radio_densidad=self.radio_densidad,
                    percentil_dist=self.percentil_dist,
                    percentil_entropia=self.percentil_entropia,
                    percentil_densidad=self.percentil_densidad,
                    criterio_pureza=self.criterio_pureza,
                    modo_espacial=self.modo_espacial,
                    factor_equilibrio=self.factor_equilibrio,
                    verbose=False
                )
                X_bin_res, y_bin_res = sampler_tmp.fit_resample(X, y_bin, max_sinteticas=faltante)
                meta_clase = getattr(sampler_tmp, "_meta", {}) or meta_clase

                nuevos = len(X_bin_res) - len(X)
                if nuevos > 0:
                    X_nuevos = X_bin_res[-nuevos:]
                    y_nuevos = np.full(nuevos, clase)
                    X_res = np.vstack([X_res, X_nuevos])
                    y_res = np.hstack([y_res, y_nuevos])

                motivo = ""
                if estado == "no se sobremuestrea":
                    motivo = "sin_faltante(actual>=objetivo)"
                elif estado == "sobremuestreada" and nuevos == 0:
                    if tope_global_aplicado:
                        motivo = "tope_global"
                    elif tope_por_clase_aplicado:
                        motivo = "tope_por_clase"
                    else:
                        cand = meta_clase.get("n_candidatas")
                        filt = meta_clase.get("n_filtradas")
                        vvp  = meta_clase.get("vecinos_validos_promedio")
                        if cand is not None and cand < (self.k + 1):
                            motivo = "insuficientes_candidatas(<k+1)"
                        elif filt is not None and filt < (self.k + 1):
                            motivo = "insuficientes_filtradas(<k+1)"
                        elif vvp is not None and vvp == 0:
                            motivo = "sin_vecinos_validos"
                        else:
                            motivo = "desconocido"
                else:
                    motivo = "ok"

            # Log por clase
            self.logs_por_clase.append({
                "dataset": self.nombre_dataset,
                "clase": int(clase) if np.issubdtype(np.array(clase).dtype, np.integer) else str(clase),
                "train_original": actual,
                "objetivo_balance": objetivo,
                "estado": estado,

                # motivos de estado
                "motivo_sin_sinteticas": motivo,
                "faltante_solicitado": int(faltante_solicitado),
                "faltante_final": int(faltante),
                "tope_por_clase_aplicado": bool(tope_por_clase_aplicado),
                "tope_global_aplicado": bool(tope_global_aplicado),
                "objetivo_alcanzado": int(actual + nuevos >= objetivo),
                "scaling_strategy": "pre_split",
              
                "muestras_sinteticas_generadas": int(nuevos),
                "total_original": total_original,
                "total_resampled": int(len(y_res)),
                "ratio_original": round(actual / total_original, 6),
                "ratio_resampled": round((actual + nuevos) / len(y_res), 6),

                # Diagnóstico del filtro (copiado de _meta del run binario)
                "n_candidatas": meta_clase.get("n_candidatas"),
                "n_filtradas": meta_clase.get("n_filtradas"),
                "riesgo_medio": meta_clase.get("riesgo_medio"),
                "riesgo_std": meta_clase.get("riesgo_std"),
                "densidad_media": meta_clase.get("densidad_media"),
                "vecinos_validos_promedio": meta_clase.get("vecinos_validos_promedio"),
                "umbral_densidad": meta_clase.get("umbral_densidad"),
                "umbral_entropia": meta_clase.get("umbral_entropia"),

                # Parámetros de referencia
                "percentil_densidad": self.percentil_densidad,
                "percentil_riesgo": self.percentil_dist,
                "criterio_pureza": self.criterio_pureza,
                "tecnica_sobremuestreo": "PCSMOTE",
                "factor_equilibrio": self.factor_equilibrio,
                "random_state": self._loggable_random_state(),
                "modo_espacial": self.modo_espacial,
                "timestamp": pd.Timestamp.now().isoformat(),

            })

        return X_res, y_res
