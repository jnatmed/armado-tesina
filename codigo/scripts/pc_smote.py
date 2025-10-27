# pcs_smote.py

""" ----- Glosario ----- 
* fallback: mecanismo de respaldo o alternativa que se utiliza cuando el método principal falla o no está disponible. 
* ridge: (no se usa en LSD) técnica de regularización que añade una penalización a la magnitud de los coeficientes en modelos de regresión para prevenir el sobreajuste. 
* pseudo-inversa: (no se usa en LSD) generalización de la matriz inversa aplicable a matrices no cuadradas o singulares. 
* semilla: muestra de un dataset original que se usa como punto de partida para generar nuevas muestras sintéticas. 
* epsilom: valor muy pequeño usado para evitar divisiones por cero o inestabilidades numéricas. 
* varianza/covarianza: (no se usan en LSD) medidas de dispersión y co-dispersión.
* traza: (no se usa en LSD) suma de la diagonal de una matriz cuadrada.
""" 

# pcs_smote.py (refactor sin LSD ni caché; con DistanceMetric)

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.metrics import DistanceMetric
from scipy.stats import entropy
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import time

from Utils import Utils  # reset/export/_log_muestra/etc.


class PCSMOTE(Utils):
    """
    PC-SMOTE (refactor):
      - Usa métricas de distancia de scikit-learn (DistanceMetric/NearestNeighbors).
      - Densidad por intersección de esferas en el subespacio minoritario.
      - Filtro por pureza (entropía o proporción), riesgo local y densidad.
      - Logs por clase y por muestra, sin caché ni LSD.

    Convenciones:
      • Binario en fit_resample: y ∈ {0,1}, 1 = clase minoritaria.
      • Multiclase vía one-vs-max en fit_resample_multiclass.
    """

    # Rango de delta (fijo “intermedio”, antes dependía de categoria_por_Nmaj)
    DELTA_RANGE_INTERMEDIO = (0.4, 0.6)

    def __init__(self,
                 k_neighbors=7,
                 random_state=None,
                 radio_densidad=1.0,          # se mantiene por compatibilidad (no se usa explícito)
                 percentil_dist=75,           # percentil para umbrales locales
                 percentil_entropia=None,     # si criterio_pureza='entropia'
                 percentil_densidad=None,     # umbral por percentil sobre densidades
                 criterio_pureza='entropia',  # 'entropia' o 'proporcion'
                 modo_espacial='2d',          # compat.
                 factor_equilibrio=0.8,
                 verbose=True,
                 max_total_multiplier=None,
                 max_sinteticas_por_clase=None,
                 guardar_distancias=True,     # compat.
                 metric='euclidean'           # NUEVO: métrica de distancia ('euclidean', 'manhattan', etc.)
                 ):
        # Hiperparámetros esenciales
        self.k = int(k_neighbors)
        self._seed_init = random_state
        self.random_state = check_random_state(random_state)

        self.radio_densidad = float(radio_densidad)
        self.percentil_dist = float(percentil_dist)
        self.percentil_entropia = None if percentil_entropia is None else float(percentil_entropia)
        self.percentil_densidad = None if percentil_densidad is None else float(percentil_densidad)
        self.criterio_pureza = str(criterio_pureza)
        self.modo_espacial = str(modo_espacial).lower()
        self.factor_equilibrio = float(factor_equilibrio)
        self.verbose = bool(verbose)
        self.guardar_distancias = bool(guardar_distancias)

        # Métrica
        self.metric = str(metric)
        self._dist_metric = DistanceMetric.get_metric(self.metric)

        # Topes
        self.max_total_multiplier = None if max_total_multiplier is None else float(max_total_multiplier)
        self.max_sinteticas_por_clase = None if max_sinteticas_por_clase is None else int(max_sinteticas_por_clase)

        # Logging y metadatos
        self.logs_por_clase = []
        self.logs_por_muestra = []
        self.meta_experimento = {}
        self._meta = {}

        # Nombre de dataset (si el flujo externo lo setea)
        self.nombre_dataset = getattr(self, "nombre_dataset", "unknown")

        # Diagnóstico opcional
        self._diag_densidad = None

    # -------------------------------
    #  Densidad por intersección
    # -------------------------------
    def calcular_densidad_interseccion(self, X_min, vecinos_local, dists_min_local):
        """
        Densidad por intersección entre semillas MINORITARIAS:
          - Para cada semilla i, define un radio local u_i como el percentil
            `percentil_dist` de las distancias a sus k vecinos minoritarios.
          - Cuenta qué fracción de esos vecinos cae a distancia <= u_i
            usando DistanceMetric.get_metric(self.metric).

        Parámetros
        ----------
        X_min : np.ndarray, shape (n_min, d)
        vecinos_local : np.ndarray[int], shape (n_min, k)
            Índices de vecinos en el índice local de X_min (excluye self).
        dists_min_local : np.ndarray[float], shape (n_min, k)
            Distancias (con la métrica elegida) de cada semilla a sus k vecinos minoritarios.

        Retorna
        -------
        np.ndarray[float], shape (n_min,)
            Densidad (proporción de intersecciones) por semilla.
        """
        X_min = np.asarray(X_min)
        n_min = len(X_min)
        if n_min == 0:
            return np.array([], dtype=float)

        densidades = np.zeros(n_min, dtype=float)
        self._diag_densidad = {"semillas_con_hits": 0, "total_hits": 0}

        for i in range(n_min):
            nbr_idx_local = vecinos_local[i]
            if len(nbr_idx_local) == 0:
                densidades[i] = 0.0
                continue

            # Umbral local por percentil sobre distancias a vecinos minoritarios
            d_i = dists_min_local[i]  # shape (k,)
            u_i = float(np.percentile(d_i, self.percentil_dist))

            # Distancias reales (recalculo con DistanceMetric para cumplir requisito)
            xi = X_min[i].reshape(1, -1)
            xj = X_min[nbr_idx_local]  # (k, d)
            dij = self._dist_metric.pairwise(xi, xj).ravel()  # (k,)

            intersecciones = int(np.sum(dij <= u_i))

            if intersecciones > 0:
                self._diag_densidad["semillas_con_hits"] += 1
                self._diag_densidad["total_hits"] += intersecciones

            densidades[i] = intersecciones / max(1, len(nbr_idx_local))

        return densidades

    def calcular_entropia(self, vecinos_all_global, y):
        """Entropía de clases en el vecindario (base 2)."""
        entropias = []
        for idxs in vecinos_all_global:
            clases, counts = np.unique(y[idxs], return_counts=True)
            p = counts / counts.sum()
            entropias.append(float(entropy(p, base=2)))
        return np.array(entropias, dtype=float)

    def _loggable_random_state(self):
        if isinstance(self._seed_init, (int, np.integer)):
            return int(self._seed_init)
        if self._seed_init is None:
            return None
        return str(self._seed_init)

    # -------------------------------
    #        FIT / RESAMPLE
    # -------------------------------
    def fit_resample(self, X, y, max_sinteticas=None):
        """
        Sobremuestreo binario (y ∈ {0,1}, 1 = minoritaria) usando:
          - Vecindarios construidos con NearestNeighbors(metric=self.metric).
          - Umbral local por percentil de distancias.
          - Densidad por intersección en subespacio minoritario.
          - Filtro por pureza (entropía o proporción).
          - Interpolación con delta ~ U[0.4, 0.6].

        Retorna
        -------
        X_resampled, y_resampled
        """
        t0 = time.perf_counter()
        X = np.asarray(X)
        y = np.asarray(y)

        self._meta = {
            "umbral_densidad": None,
            "umbral_entropia": None,
            "riesgo_medio": None,
            "riesgo_std": None,
            "densidad_media": None,
            "vecinos_validos_promedio": None,
            "n_candidatas": None,
            "n_filtradas": None,
            "elapsed_ms": None
        }

        # Separación de clases
        idxs_min_global = np.where(y == 1)[0]
        idxs_maj_global = np.where(y == 0)[0]
        X_min = X[idxs_min_global]
        X_maj = X[idxs_maj_global]

        # Necesitamos al menos k+1 (contando self) para poder excluir self y quedarnos con k
        if len(X_min) < self.k + 1:
            self._meta.update({
                "n_candidatas": int(len(X_min)),
                "n_filtradas": 0,
                "elapsed_ms": (time.perf_counter() - t0) * 1000
            })
            return X.copy(), y.copy()

        K = int(self.k)

        # -------------------------------------------
        # Vecindarios con la métrica elegida
        # -------------------------------------------
        # Global (vecinos mixtos) — consultado SOLO para semillas minoritarias
        nn_global = NearestNeighbors(n_neighbors=K + 1, metric=self.metric).fit(X)
        d_all, i_all = nn_global.kneighbors(X_min, return_distance=True)   # incluye self en la 1ª col
        d_all = d_all[:, 1:]   # (n_min, K)
        i_all = i_all[:, 1:]   # (n_min, K) índices globales (en X)

        # Minoritaria local (vecinos de la MISMA clase)
        nn_min = NearestNeighbors(n_neighbors=K + 1, metric=self.metric).fit(X_min)
        d_min, i_min_local = nn_min.kneighbors(X_min, return_distance=True)
        d_min = d_min[:, 1:]               # (n_min, K)
        i_min_local = i_min_local[:, 1:]   # (n_min, K) índices locales (en X_min)

        vecinos_all_global = i_all                   # (n_min, K) — índices en X
        vecinos_min_local = i_min_local              # (n_min, K) — índices en X_min

        # Vecinos minoritarios en índices globales (para logging/coherencia con flujo anterior)
        vecinos_min_global = np.array([
            idxs_min_global[row] for row in vecinos_min_local
        ], dtype=object)

        # -------------------------------------------
        # Métricas locales
        # -------------------------------------------
        # Riesgo: proporción de vecinos mayoritarios (sobre K vecinos globales)
        riesgo = np.array([
            np.sum(y[idxs] == 0) / K for idxs in vecinos_all_global
        ], dtype=float)

        # Densidad por intersección en minoritaria (usa DistanceMetric internamente)
        densidades = self.calcular_densidad_interseccion(
            X_min=X_min,
            vecinos_local=vecinos_min_local,
            dists_min_local=d_min
        )

        # Pureza (entropía o proporción minoritaria entre los K vecinos globales)
        pureza_mask = None
        umb_ent = None
        entropias = None
        proporciones_min = None

        if self.criterio_pureza == 'entropia':
            entropias = self.calcular_entropia(vecinos_all_global, y)
            umb_ent = float(np.percentile(entropias, self.percentil_entropia)) if self.percentil_entropia else None
            pureza_mask = entropias <= (umb_ent if umb_ent is not None else 1.0)
            self._meta["umbral_entropia"] = umb_ent
        elif self.criterio_pureza == 'proporcion':
            proporciones_min = np.array([
                np.sum(y[idxs] == 1) / K for idxs in vecinos_all_global
            ], dtype=float)
            # ventana 40–60% minoritaria
            pureza_mask = (proporciones_min >= 0.4) & (proporciones_min <= 0.6)
        else:
            raise ValueError(f"Criterio de pureza no reconocido: {self.criterio_pureza}")

        # Filtro por densidad
        if self.percentil_densidad is not None:
            umb_den = float(np.percentile(densidades, self.percentil_densidad))
            densidad_mask = densidades >= umb_den
            self._meta["umbral_densidad"] = umb_den
        else:
            umb_den = None
            densidad_mask = densidades > 0.0

        # Intersección de criterios
        comb = pureza_mask & densidad_mask
        filtered_indices_local = np.where(comb)[0]                 # en X_min
        filtered_indices_global = idxs_min_global[filtered_indices_local]  # en X

        # Métricas agregadas
        self._meta.update({
            "n_candidatas": int(len(X_min)),
            "n_filtradas": int(np.sum(comb)),
            "riesgo_medio": float(np.mean(riesgo[comb])) if np.any(comb) else None,
            "riesgo_std": float(np.std(riesgo[comb])) if np.any(comb) else None,
            "densidad_media": float(np.mean(densidades)) if densidades.size else None
        })

        # -------------------------------------------
        # Diagnóstico de vecinos válidos
        # (percentil sobre distancias a K vecinos globales)
        # -------------------------------------------
        dist_thr_por_muestra = np.percentile(d_all, self.percentil_dist, axis=1).astype(float)
        vecinos_validos_counts = np.sum(d_all <= dist_thr_por_muestra[:, None], axis=1).astype(int)
        self._meta["vecinos_validos_promedio"] = float(np.mean(vecinos_validos_counts)) if len(vecinos_validos_counts) else 0.0

        # Cortes por insuficiencia
        if len(filtered_indices_local) < (K + 1):
            self._registrar_logs_sin_sinteticas(
                X, y, X_min, idxs_min_global,
                comb, riesgo, densidades, entropias, proporciones_min,
                pureza_mask, densidad_mask,
                umb_ent, umb_den,
                vecinos_all_global, vecinos_min_global,
                vecinos_validos_counts, dist_thr_por_muestra
            )
            self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            return X.copy(), y.copy()

        n_sint = int(max_sinteticas if max_sinteticas is not None else len(X_maj) - len(X_min))
        if n_sint <= 0:
            self._registrar_logs_sin_sinteticas(
                X, y, X_min, idxs_min_global,
                comb, riesgo, densidades, entropias, proporciones_min,
                pureza_mask, densidad_mask,
                umb_ent, umb_den,
                vecinos_all_global, vecinos_min_global,
                vecinos_validos_counts, dist_thr_por_muestra
            )
            self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            return X.copy(), y.copy()

        # -------------------------------------------
        # Precálculos recortados a semillas filtradas
        # -------------------------------------------
        X_min_filtrado = X_min[filtered_indices_local]
        vecinos_all_filtrado = vecinos_all_global[filtered_indices_local]   # (n_filtradas, K) índices en X
        dists_filtradas = d_all[filtered_indices_local]                    # (n_filtradas, K)
        thr_filtradas = dist_thr_por_muestra[filtered_indices_local]       # (n_filtradas,)
        # riesgo_filt = riesgo[filtered_indices_local]  # ya no se usa para delta

        # Estructuras de logging
        gen_from_counts = defaultdict(int)
        last_delta_by_seed = {}
        last_neighbor_by_seed = {}

        rng = self.random_state
        # Rango fijo de delta
        # por defecto lo es 0.4, hi es 0.6
        lo, hi = self.DELTA_RANGE_INTERMEDIO

        # -------------------------------------------
        # Generación de sintéticas
        # -------------------------------------------
        muestras_sinteticas = []

        for _ in range(n_sint):
            # Elegimos una semilla de las filtradas
            idx_loc = int(rng.randint(len(X_min_filtrado)))
            xi = X_min_filtrado[idx_loc]

            # Filtramos vecinos válidos por percentil (globales)
            idxs_vec_all = vecinos_all_filtrado[idx_loc]  # índices en X
            dists = dists_filtradas[idx_loc]              # distancias correspondientes
            thr = thr_filtradas[idx_loc]

            vecinos_validos = idxs_vec_all[dists <= thr]
            if len(vecinos_validos) == 0:
                continue

            z_idx = int(rng.choice(vecinos_validos))
            xz = X[z_idx]

            delta = float(rng.uniform(lo, hi))
            # formula de interpolación
            # xi + delta * (xz - xi)
            # la nueva muestra se acerca más al vecino elegido
            muestras_sinteticas.append(xi + delta * (xz - xi))

            # Logging por semilla (índice global de la semilla original)
            seed_global_idx = int(filtered_indices_global[idx_loc])
            gen_from_counts[seed_global_idx] += 1
            last_delta_by_seed[seed_global_idx] = delta
            last_neighbor_by_seed[seed_global_idx] = z_idx

        # Conclusión
        if not muestras_sinteticas:
            self._registrar_logs_sin_sinteticas(
                X, y, X_min, idxs_min_global,
                comb, riesgo, densidades, entropias, proporciones_min,
                pureza_mask, densidad_mask,
                umb_ent, umb_den,
                vecinos_all_global, vecinos_min_global,
                vecinos_validos_counts, dist_thr_por_muestra
            )
            self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            return X.copy(), y.copy()

        X_sint = np.asarray(muestras_sinteticas, dtype=float)
        y_sint = np.ones(len(X_sint), dtype=int)

        X_resampled = np.vstack([X, X_sint])
        y_resampled = np.hstack([y, y_sint])

        # Log por muestra de la minoritaria original
        for i in range(len(X_min)):
            self._log_muestra(
                i, X, X_min, y, idxs_min_global,
                comb, riesgo, densidades, entropias, proporciones_min,
                pureza_mask, densidad_mask,
                umb_ent, umb_den,
                vecinos_all_global, vecinos_min_global,
                vecinos_validos_counts, dist_thr_por_muestra,
                gen_from_counts, last_delta_by_seed, last_neighbor_by_seed
            )

        self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        return X_resampled, y_resampled

    # ------------------------------------------
    #        Multiclase (one-vs-max)
    # ------------------------------------------
    def fit_resample_multiclass(self, X, y):
        """
        Oversampling por clase contra la mayor, respetando factor_equilibrio
        y topes globales/por clase. Reusa el nuevo binario (sin LSD ni caché).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        clases = np.unique(y)
        X_res = X.copy()
        y_res = y.copy()

        total_original = len(y)
        conteo_original = Counter(y)
        max_count = max(conteo_original.values())

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
            "metric": self.metric,
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

            tope_por_clase_aplicado = False
            tope_global_aplicado = False

            # Tope por clase
            if self.max_sinteticas_por_clase is not None:
                nuevo_faltante = min(faltante, self.max_sinteticas_por_clase)
                tope_por_clase_aplicado = (nuevo_faltante < faltante)
                faltante = nuevo_faltante

            # Tope global
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
                    verbose=False,
                    max_total_multiplier=None,
                    max_sinteticas_por_clase=None,
                    guardar_distancias=self.guardar_distancias,
                    metric=self.metric
                )
                sampler_tmp.nombre_dataset = self.nombre_dataset

                X_bin_res, y_bin_res = sampler_tmp.fit_resample(X, y_bin, max_sinteticas=faltante)
                meta_clase = getattr(sampler_tmp, "_meta", {}) or meta_clase

                nuevos = len(X_bin_res) - len(X)
                if nuevos > 0:
                    X_nuevos = X_bin_res[-nuevos:]
                    y_nuevos = np.full(nuevos, clase)
                    X_res = np.vstack([X_res, X_nuevos])
                    y_res = np.hstack([y_res, y_nuevos])

                # Copiar logs por muestra agregando clase_objetivo
                for rec in sampler_tmp.logs_por_muestra:
                    rec_copia = dict(rec)
                    rec_copia["clase_objetivo"] = clase
                    keys = list(rec_copia.keys())
                    if "idx_global" in keys and "clase_objetivo" in keys:
                        k_cls = keys.pop(keys.index("clase_objetivo"))
                        keys.insert(keys.index("idx_global") + 1, k_cls)
                        rec_copia = {k: rec_copia[k] for k in keys}
                    self.logs_por_muestra.append(rec_copia)

                # Motivo
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
            else:
                motivo = "sin_faltante(actual>=objetivo)" if estado != "sobremuestreada" else "tope=0"

            # Log por clase (resumen)
            self.logs_por_clase.append({
                "dataset": self.nombre_dataset,
                "clase": int(clase) if np.issubdtype(np.array(clase).dtype, np.integer) else str(clase),
                "train_original": actual,
                "objetivo_balance": objetivo,
                "estado": estado,
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
                "ratio_original": round(actual / total_original, 6) if total_original else None,
                "ratio_resampled": round((actual + nuevos) / len(y_res), 6) if len(y_res) else None,
                "n_candidatas": meta_clase.get("n_candidatas"),
                "n_filtradas": meta_clase.get("n_filtradas"),
                "riesgo_medio": meta_clase.get("riesgo_medio"),
                "riesgo_std": meta_clase.get("riesgo_std"),
                "densidad_media": meta_clase.get("densidad_media"),
                "vecinos_validos_promedio": meta_clase.get("vecinos_validos_promedio"),
                "umbral_densidad": meta_clase.get("umbral_densidad"),
                "umbral_entropia": meta_clase.get("umbral_entropia"),
                "percentil_densidad": self.percentil_densidad,
                "percentil_riesgo": self.percentil_dist,
                "criterio_pureza": self.criterio_pureza,
                "tecnica_sobremuestreo": "PCSMOTE",
                "factor_equilibrio": self.factor_equilibrio,
                "random_state": self._loggable_random_state(),
                "modo_espacial": self.modo_espacial,
                "metric": self.metric,
                "timestamp": pd.Timestamp.now().isoformat(),
            })

        return X_res, y_res
