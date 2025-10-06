from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from scipy.stats import entropy
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import json
import time


class PCSMOTE:
    """
    PC-SMOTE con:
      - Topes de crecimiento (global y por clase).
      - Log por CLASE (resumen) y log POR MUESTRA (detalle).
      - Exportaciones CSV/JSON.
      - Opción para guardar distancias a vecinos.

    Notas:
      • fit_resample: binario (y ∈ {0,1}, 1 = minoritaria).
      • fit_resample_multiclass: itera clase por clase contra la mayor.
    """
    umbral_distancia = None # umbral global (float)


    def __init__(self, k_neighbors=5, random_state=None,
                 radio_densidad=1.0, percentil_dist=75,
                 percentil_entropia=None, percentil_densidad=None,
                 criterio_pureza='entropia', modo_espacial='2d',
                 factor_equilibrio=0.8, verbose=True,
                 max_total_multiplier=None,
                 max_sinteticas_por_clase=None,
                 guardar_distancias=True):  # ← por defecto True
        # Hiperparámetros
        self.k = int(k_neighbors) # cantodad de k vecinos mas cercanos a examinar
        self._seed_init = random_state # semilla fijada para reproductividad y mantener consistencias entre corridas
        self.random_state = check_random_state(random_state)
        self.radio_densidad = float(radio_densidad)
        self.percentil_dist = float(percentil_dist)
        self.percentil_entropia = None if percentil_entropia is None else float(percentil_entropia)
        self.percentil_densidad = None if percentil_densidad is None else float(percentil_densidad)
        self.criterio_pureza = str(criterio_pureza)
        self.modo_espacial = str(modo_espacial).lower()  # '2d' o '3d'
        self.factor_equilibrio = float(factor_equilibrio)
        self.verbose = bool(verbose)
        self.guardar_distancias = bool(guardar_distancias)

        # Topes
        self.max_total_multiplier = None if max_total_multiplier is None else float(max_total_multiplier)
        self.max_sinteticas_por_clase = None if max_sinteticas_por_clase is None else int(max_sinteticas_por_clase)

        # Logging
        self.logs_por_clase = []      # resumen por clase
        self.logs_por_muestra = []    # detalle por muestra
        self.meta_experimento = {}
        self._meta = {}               # métricas del último fit_resample

        # Nombre del dataset (opcional)
        self.nombre_dataset = getattr(self, "nombre_dataset", "unknown")

    # --------------------- Utilidades ---------------------

    def reset_logs(self):
        """Resetea logs por CLASE, por MUESTRA y metadatos."""
        self.logs_por_clase = []
        self.logs_por_muestra = []
        self.meta_experimento = {}
        self._meta = {}

    def _loggable_random_state(self):
        if isinstance(self._seed_init, (int, np.integer)):
            return int(self._seed_init)
        if self._seed_init is None:
            return None
        return str(self._seed_init)

    def exportar_log_csv(self, path_salida):
        """Exporta el log por CLASE."""
        if not self.logs_por_clase:
            print("⚠️ No hay log POR CLASE para exportar.")
            return
        pd.DataFrame(self.logs_por_clase).to_csv(path_salida, index=False)
        print(f"📁 Log por clase guardado en: {path_salida}")

    def exportar_log_muestras_csv(self, path_salida):
        """Exporta el log POR MUESTRA."""
        if not self.logs_por_muestra:
            print("⚠️ No hay log POR MUESTRA para exportar.")
            return
        df = pd.DataFrame(self.logs_por_muestra)

        # Forzar orden estable de columnas (si existen)
        cols_order = [
            "dataset","idx_global","clase_objetivo","is_filtrada","k",
            "percentil_dist","percentil_densidad","percentil_entropia",
            "criterio_pureza","modo_espacial","radio_densidad",
            "riesgo","densidad","entropia","proporcion_min",
            "pasa_pureza","pasa_densidad","umbral_entropia","umbral_densidad",
            "vecinos_all","clase_vecinos_all","dist_all",
            "vecinos_min","dist_vecinos_min",
            "vecinos_validos_por_percentil","thr_dist_percentil",
            "synthetics_from_this_seed","last_delta","last_neighbor_z","timestamp"
        ]
        df = df.reindex(columns=[c for c in cols_order if c in df.columns])

        # Serializar listas en JSON para columnas complejas
        cols_json = (
            "vecinos_all", "clase_vecinos_all", "dist_all",
            "vecinos_min", "dist_vecinos_min"
        )
        for col in cols_json:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, tuple, np.ndarray))
                    else v
                )
        df.to_csv(path_salida, index=False)
        print(f"📁 Log por muestra guardado en: {path_salida}")

    def exportar_log_json_meta(self, path_json):
        if not self.meta_experimento:
            print("⚠️ No hay metadatos de experimento para exportar.")
            return
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(self.meta_experimento, f, ensure_ascii=False, indent=2)
        print(f"📁 Metadatos de experimento guardados en: {path_json}")

    # --------------------- Cálculos auxiliares ---------------------

    def _dist(self, A, b):
        """Distancia euclídea 2D/3D según modo_espacial."""
        if self.modo_espacial == '3d':
            return np.linalg.norm(A[:, :3] - b[:3], axis=1)
        return np.linalg.norm(A - b, axis=1)

    def getUmbralDistancia(self):
        return 0.0 if self.umbral_distancia is None else float(self.umbral_distancia)

    def distancia_x_mahalanobis(self, X_min, vecinos_min_local, percentil=25.0):
        """
        Calcula un umbral GLOBAL de distancia usando Mahalanobis LOCAL por vecindario.
        Guarda el resultado en self.umbral_distancia (float).
        La idea es saber que tan cerca estan los vecinos en alta dimension
        Con features correlacionadas o desbalance de escalas, la euclídea puede engañar; 
        Mahalanobis reescala por la covarianza local y captura mejor la geometría del vecindario.        
        """
        X_min = np.asarray(X_min)                       # Asegura que X_min sea un arreglo NumPy

        n_min, n_feat = X_min.shape                     # (n_min: cantidad de minoritarias, n_feat: nº de características) = X_min.shape
        todas = []                                      # Acumulará todas las distancias de Mahalanobis (de todos los xi)

        """
        En cada iteración se toma la muestra minoritaria x_i, se recuperan sus k vecinos
        minoritarios y se organizan en una matriz; x_i se convierte a fila para operar
        vectorizadamente; se estima la covarianza local S de esos vecinos y se regulariza
        sumando λI para estabilizar la inversión; se calcula una pseudo-inversa estable
        de S_reg y, con ella, se obtienen las distancias de Mahalanobis de cada vecino
        a x_i, que se agregan a un acumulador global. Si ocurre algún problema numérico,
        se usa un fallback a distancias euclídeas y también se acumulan. Al finalizar
        todas las iteraciones, ese conjunto de distancias servirá para fijar un umbral
        global por percentil que defina la “cercanía” en la métrica.
        """
        for i in range(n_min):                          # Itera por cada semilla minoritaria xi
            idx_nbrs = vecinos_min_local[i]             # Índices locales de los k vecinos minoritarios de xi
            if len(idx_nbrs) == 0:                      # Si una xi no tiene vecinos, salta
                continue

            # ---- Datos de los vecinos ----
            # nbrs = neighbors = vecinos    
            nbrs = X_min[idx_nbrs]                      # Matriz (k, n_feat) con los vecinos de xi
            
            xi = X_min[i].reshape(1, -1)                # Xi como fila (1, n_feat) para restar vectorizado

            try:
                # ---- Covarianza local de los vecinos ----
                if nbrs.shape[0] == 1:                  # Con un solo vecino la covarianza no es válida
                    S = np.eye(n_feat, dtype=float)     # Usa identidad como covarianza (fallback estable)
                else:
                    S = np.cov(nbrs, rowvar=False)      # Covarianza columna-variable (n_feat x n_feat)
                    if np.ndim(S) == 0:                 # Si sale degenerada (escala), usa identidad
                        S = np.eye(n_feat, dtype=float)

                # ---- Regularización (ridge) para evitar singularidades ----
                """
                - 1e-6, es un ε (epsilon) pequeño 
                - trace(S) = suma de las varianzas (la traza de la covarianza).
                - n_feat = p, el número de features (dimensión).
                - trace(S)/p = promedio de varianzas (la “escala” típica)
                - max(1, n_feat) es un guardarraíl para no dividir por 0 en casos patológicos.
                En la práctica n_feat ≥ 1, pero este max asegura que si por error llegara n_feat = 0, dividas por 1 y no reviente
                """
                lam = 1e-6 * (np.trace(S) / max(1, n_feat)) # λ (lambda) proporcional a la escala de S (estable numéricamente)

                """
                - S: la matriz de covarianza estimada en el vecindario de Dimensión p x p (siendo p = n_feat).
                  Diagonal = varianzas de cada feature; fuera de diagonal = covarianzas entre features.
                - lam = λ (lambda): el parámetro de regularización (un escalar pequeño y positivo).
                        Se suma en la diagonal para estabilizar / invertir = S + λI.
                        En tu código: lam = 1e-6 * (trace(S)/p) (escala la magnitud de 
                        S y usa un ε pequeño).
                - I: la matriz identidad de tamaño p x p (unos en la diagonal, ceros fuera).
                     En NumPy: np.eye(p).                        
                Ejemplo:
                    S = np.array([[1.0, 0.2, 0.3],
                                [0.2, 2.0, 0.1],
                                [0.3, 0.1, 3.0]])
                    S.shape      # (3, 3)
                    S.shape[0]  # 3  -> filas (p)
                    S.shape[1]  # 3  -> columnas (p)                     
                    np.eye(S.shape[0], dtype=float) = np.eye(3) = np.array([[1., 0., 0.],
                                                                          [0., 1., 0.],
                                                                          [0., 0., 1.]])
                """
                S_reg = S + lam * np.eye(S.shape[0], dtype=float)  # S regularizada: S + λI

                """
                ---- Inversa estable (pseudo-inversa con tolerancia) ----
                   np.linalg.pinv(A, rcond=...) calcula la pseudo-inversa de Moore–Penrose vía SVD:
                   A = U Σ V^T  ⇒  pinv(A) = V Σ^+ U^T
                   donde Σ^+ invierte solo los σ_i “grandes” y pone 0 a los σ_i pequeños.
                   rcond: (relative condition) umbral relativo para truncar valores singulares:
                   - Si σ_i ≤ rcond * σ_max  ⇒  ese σ_i se considera ~0 (no se invierte) → mayor estabilidad.
                   - Valores típicos: 1e-15 … 1e-8; 1e-12 es un corte conservador y suele funcionar bien.
                   S_inv: pseudo-inversa de S_reg; se usa en Mahalanobis: (x − μ)^T · S_inv · (x − μ).
                """
                # ---- Inversa estable (pseudo-inversa con tolerancia) ----
                S_inv = np.linalg.pinv(S_reg, rcond=1e-12)  # Pseudo-inversa robusta ante S mal condicionada

                # ---- Distancias de Mahalanobis de cada vecino respecto a xi ----
                diffs = nbrs - xi                       # Diferencias (k, n_feat)
                # d_M(xj, xi) = sqrt( (xj-xi)^T S_inv (xj-xi) ) para cada vecino
                d_maha = np.sqrt(np.einsum('ij,jk,ik->i', diffs, S_inv, diffs))  # Vector de k distancias

                if d_maha.size:                         # Si hay distancias válidas
                    todas.append(d_maha)                # Acumula para el percentil global

            except Exception as e:
                # Fallback local: si falla Mahalanobis (cov/inversa/einsum), usar Euclídea
                try:
                    d_euc = np.linalg.norm(nbrs - xi, axis=1)
                    if d_euc.size:
                        todas.append(d_euc)
                    if getattr(self, "verbose", False):
                        print(f"[PCSMOTE][warn] Mahalanobis falló en i={i}. Fallback a euclídea. Detalle: {e}")
                except Exception as e2:
                    if getattr(self, "verbose", False):
                        print(f"[PCSMOTE][warn] Euclídea también falló en i={i}. Se omite muestra. Detalle: {e2}")
                    continue

        if len(todas) == 0:                             # Si no se pudo calcular nada (sin vecinos en todo el set)
            self.umbral_distancia = 0.0                 # Umbral neutro (no contará “cercanos”)
            return

        try:
            todas = np.concatenate(todas)               # Aplana todas las distancias (n_min*k,)
            # Percentil global (p.ej. 25): define qué tan “cerca” es ser vecino en todo el conjunto
            self.umbral_distancia = float(np.percentile(todas, percentil))  # Guarda umbral en el atributo de la clase
        except Exception as e:
            # Fallback global: si falla el percentil por algún motivo, usar 0.0
            if getattr(self, "verbose", False):
                print(f"[PCSMOTE][warn] Percentil global falló. Umbral=0.0. Detalle: {e}")
            self.umbral_distancia = 0.0



    def calcular_densidad_interseccion(self, X_min, vecinos_local, radio):
        """
        Densidad por intersección de esferas de radio 'radio', usando índices locales de X_min.
        """
        densidades = []
        umbral_distancia = self.getUmbralDistancia()  # umbral global (float)        

        for i, xi in enumerate(X_min):
            intersecciones = 0
            for j in vecinos_local[i]:
                xj = X_min[j]
                # revisar calculo de distancia. Es distancia ?
                # mejor una calculo d distancia comun. 
                # d = np.linalg.norm(xi[:3] - xj[:3]) if self.modo_espacial == '3d' else np.linalg.norm(xi - xj)
                d = np.linalg.norm(xi - xj) 
                # q intrercciones no sea mas grande q vecinos locales
                # como son las distancias ? en los X_min y saco una regla empirica
                if d <= umbral_distancia:
                    intersecciones += 1
            densidades.append(intersecciones / max(1, len(vecinos_local[i])))
        return np.array(densidades, dtype=float)

    def calcular_entropia(self, vecinos_all_global, y):
        """Entropía de clases en el vecindario (base 2)."""
        entropias = []
        for idxs in vecinos_all_global:
            clases, counts = np.unique(y[idxs], return_counts=True)
            p = counts / counts.sum()
            entropias.append(float(entropy(p, base=2)))
        return np.array(entropias, dtype=float)

    # --------------------- Logger por muestra ---------------------

    @staticmethod
    def _to_cls_scalar(v):
        """Convierte etiqueta a tipo serializable estable (int si es entero; si no, str)."""
        try:
            arr = np.array(v)
            if np.issubdtype(arr.dtype, np.integer):
                return int(arr.item() if arr.shape == () else v)
        except Exception:
            pass
        try:
            return v.item()
        except Exception:
            return str(v)

    def _log_muestra(
        self,
        i,                      # índice en X_min
        X, X_min,               # matrices originales y minoritaria
        y,                      # etiquetas globales (para clases de vecinos)
        idxs_min_global,        # mapeo X_min[i] -> índice global en X
        comb,                   # máscara de filtrado por muestra
        riesgo, densidades,     # arrays
        entropias, proporciones_min,  # arrays o None
        pureza_mask, densidad_mask,   # máscaras booleanas
        umb_ent, umb_den,       # umbrales (float o None)
        vecinos_all_global,     # [n_min, k] índices globales en X
        vecinos_min_global,     # [n_min, k] índices globales minoritarios
        vecinos_validos_counts, # array de conteos válidos por percentil_dist
        dist_thr_por_muestra,   # array thresholds por muestra
        gen_from_counts,        # dict: idx_global -> sintéticas desde esa semilla
        last_delta_by_seed,     # dict: idx_global -> último delta
        last_neighbor_by_seed   # dict: idx_global -> último vecino z (idx global)
    ):
        seed_idx_global = int(idxs_min_global[i])

        # Vecinos (globales)
        v_all = list(map(int, vecinos_all_global[i].tolist()))
        v_min = list(map(int, vecinos_min_global[i].tolist()))
        # Clases de vecinos_all
        cls_all = [self._to_cls_scalar(y[idx]) for idx in v_all]

        # Distancias (opcionales)
        if self.guardar_distancias:
            xi = X_min[i]
            d_all = self._dist(X[v_all], xi).tolist() if len(v_all) else []
            d_min = self._dist(X[v_min], xi).tolist() if len(v_min) else []
            d_vecinos_min = d_min[:]  # alias explícito pedido
        else:
            d_all = None
            d_min = None
            d_vecinos_min = None

        rec = {
            "dataset": self.nombre_dataset,
            "idx_global": seed_idx_global,
            "clase_objetivo": None,  # ← por defecto; se completa en multiclase
            "is_filtrada": bool(comb[i]),
            "k": self.k,
            "percentil_dist": self.percentil_dist,
            "percentil_densidad": self.percentil_densidad,
            "percentil_entropia": self.percentil_entropia,
            "criterio_pureza": self.criterio_pureza,
            "modo_espacial": self.modo_espacial,
            "radio_densidad": self.radio_densidad,
            "riesgo": float(riesgo[i]),
            "densidad": float(densidades[i]),
            "entropia": None if entropias is None else float(entropias[i]),
            "proporcion_min": None if proporciones_min is None else float(proporciones_min[i]),
            "pasa_pureza": bool(pureza_mask[i]),
            "pasa_densidad": bool(densidad_mask[i]),
            "umbral_entropia": umb_ent,
            "umbral_densidad": umb_den,
            # Vecinos y distancias
            "vecinos_all": v_all,
            "clase_vecinos_all": cls_all,
            "dist_all": d_all,
            "vecinos_min": v_min,
            "dist_vecinos_min": d_vecinos_min,
            # Diagnóstico percentil de distancia
            "vecinos_validos_por_percentil": int(vecinos_validos_counts[i]),
            "thr_dist_percentil": float(dist_thr_por_muestra[i]),
            # Uso en síntesis
            "synthetics_from_this_seed": int(gen_from_counts.get(seed_idx_global, 0)),
            "last_delta": last_delta_by_seed.get(seed_idx_global, None),
            "last_neighbor_z": last_neighbor_by_seed.get(seed_idx_global, None),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self.logs_por_muestra.append(rec)

    # --------------------- Núcleo binario ---------------------

    def fit_resample(self, X, y, max_sinteticas=None):
        """
        Resample binario: y ∈ {0,1} (1 = minoritaria).
        Devuelve (X_resampled, y_resampled).
        Registra LOG POR MUESTRA y guarda métricas agregadas en self._meta.
        """
        t0 = time.perf_counter()

        X = np.asarray(X)
        y = np.asarray(y)

        # Inicializar meta
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

        # Separación binaria
        idxs_min_global = np.where(y == 1)[0]
        idxs_maj_global = np.where(y == 0)[0]
        X_min = X[idxs_min_global]
        X_maj = X[idxs_maj_global]

        # No alcanza para vecinos
        if len(X_min) < self.k + 1:
            self._meta.update({"n_candidatas": int(len(X_min)), "n_filtradas": 0, "elapsed_ms": (time.perf_counter()-t0)*1000})
            return X.copy(), y.copy()

        # Vecinos para riesgo (todo X) y densidad (solo minoritaria)
        nn_all = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        vecinos_all_global = nn_all.kneighbors(X_min, return_distance=False)[:, 1:]  # índices globales

        riesgo = np.array([np.sum(y[idxs] == 0) / self.k for idxs in vecinos_all_global], dtype=float)

        nn_min = NearestNeighbors(n_neighbors=self.k + 1).fit(X_min)
        vecinos_min_local = nn_min.kneighbors(X_min, return_distance=False)[:, 1:]  # índices locales de X_min
        # mapear a global
        vecinos_min_global = np.array([[int(idxs_min_global[j]) for j in fila] for fila in vecinos_min_local], dtype=int)

        # los vecinos locales de cada muestra minoritaria los envio en 
        # vecinos_min_local
        # junto con X_min
        # de manera que 
        # Calcula y guarda el umbral global de distancia (Mahalanobis local)
        self.distancia_x_mahalanobis(
            X_min=X_min,
            vecinos_min_local=vecinos_min_local,
            percentil=self.percentil_dist  # o un percentil específico si preferís
        )

        densidades = self.calcular_densidad_interseccion(X_min, vecinos_min_local, self.radio_densidad)

        # Pureza
        pureza_mask = None
        umb_ent = None
        entropias = None
        proporciones_min = None

        if self.criterio_pureza == 'entropia':
            entropias = self.calcular_entropia(vecinos_all_global, y)
            umb_ent = float(np.percentile(entropias, self.percentil_entropia)) if self.percentil_entropia is not None else None
            pureza_mask = entropias <= (umb_ent if umb_ent is not None else 1.0)
            self._meta["umbral_entropia"] = umb_ent
        elif self.criterio_pureza == 'proporcion':
            proporciones_min = np.array([np.sum(y[idxs] == 1) / self.k for idxs in vecinos_all_global], dtype=float)
            # frontera ~ [0.4, 0.6]
            pureza_mask = (proporciones_min >= 0.4) & (proporciones_min <= 0.6)
        else:
            raise ValueError(f"Criterio de pureza no reconocido: {self.criterio_pureza}")

        # Densidad
        if self.percentil_densidad is not None:
            umb_den = float(np.percentile(densidades, self.percentil_densidad))
            densidad_mask = densidades >= umb_den
            self._meta["umbral_densidad"] = umb_den
        else:
            umb_den = None
            densidad_mask = densidades > 0.0

        # Combinación
        comb = pureza_mask & densidad_mask
        filtered_indices_local = np.where(comb)[0]                         # en X_min
        filtered_indices_global = idxs_min_global[filtered_indices_local]  # en X

        # Métricas agregadas
        self._meta.update({
            "n_candidatas": int(len(X_min)),
            "n_filtradas": int(np.sum(comb)),
            "riesgo_medio": float(np.mean(riesgo[comb])) if np.any(comb) else None,
            "riesgo_std": float(np.std(riesgo[comb])) if np.any(comb) else None,
            "densidad_media": float(np.mean(densidades)) if densidades.size else None
        })

        # Vecinos válidos por percentil_dist (diagnóstico)
        vecinos_validos_counts = np.zeros(len(X_min), dtype=int)
        dist_thr_por_muestra = np.full(len(X_min), np.nan)
        for i in range(len(X_min)):
            idxs_vec_all = vecinos_all_global[i]  # globales
            xi = X_min[i]
            dists = self._dist(X[idxs_vec_all], xi)
            thr = np.percentile(dists, self.percentil_dist)
            dist_thr_por_muestra[i] = float(thr)
            vecinos_validos_counts[i] = int(np.sum(dists <= thr))
        self._meta["vecinos_validos_promedio"] = float(np.mean(vecinos_validos_counts)) if len(vecinos_validos_counts) else None

        # Contadores por semilla (para log posterior)
        gen_from_counts = defaultdict(int)
        last_delta_by_seed = {}
        last_neighbor_by_seed = {}

        # Salidas tempranas
        if len(filtered_indices_local) < self.k + 1:
            for i in range(len(X_min)):
                self._log_muestra(
                    i, X, X_min, y, idxs_min_global,
                    comb, riesgo, densidades,
                    entropias, proporciones_min,
                    pureza_mask, densidad_mask,
                    umb_ent, None if self.percentil_densidad is None else float(self._meta["umbral_densidad"]),
                    vecinos_all_global, vecinos_min_global,
                    vecinos_validos_counts, dist_thr_por_muestra,
                    {}, {}, {}
                )
            self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            return X.copy(), y.copy()

        n_sint = max_sinteticas if max_sinteticas is not None else (len(X_maj) - len(X_min))
        n_sint = int(max(0, n_sint))
        # sino se produjeron sinteticas, entonces logueo y salgo
        if n_sint == 0:
            for i in range(len(X_min)):
                self._log_muestra(
                    i, X, X_min, y, idxs_min_global,
                    comb, riesgo, densidades,
                    entropias, proporciones_min,
                    pureza_mask, densidad_mask,
                    umb_ent, None if self.percentil_densidad is None else float(self._meta["umbral_densidad"]),
                    vecinos_all_global, vecinos_min_global,
                    vecinos_validos_counts, dist_thr_por_muestra,
                    {}, {}, {}
                )
            self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            return X.copy(), y.copy()

        # Generación de sintéticas
        X_min_filtrado = X_min[filtered_indices_local]
        vecinos_all_filtrado = vecinos_all_global[filtered_indices_local]
        muestras_sinteticas = []

        for _ in range(n_sint):
            idx_local_filt = self.random_state.randint(len(X_min_filtrado))
            xi = X_min_filtrado[idx_local_filt]
            r_i = riesgo[filtered_indices_local][idx_local_filt]
            idxs_vec_all = vecinos_all_filtrado[idx_local_filt]  # globales

            dists = self._dist(X[idxs_vec_all], xi)
            thr = np.percentile(dists, self.percentil_dist)
            vecinos_validos = idxs_vec_all[dists <= thr]
            if len(vecinos_validos) == 0:
                continue

            z_idx = int(self.random_state.choice(vecinos_validos))
            xz = X[z_idx]

            # Delta según riesgo local
            if 0.4 <= r_i < 0.5:
                delta = float(self.random_state.uniform(0.6, 0.8))
            elif 0.5 <= r_i <= 0.6:
                delta = float(self.random_state.uniform(0.3, 0.5))
            else:
                delta = float(self.random_state.uniform(0.4, 0.6))

            muestras_sinteticas.append(xi + delta * (xz - xi))

            seed_global_idx = int(filtered_indices_global[idx_local_filt])
            gen_from_counts[seed_global_idx] += 1
            last_delta_by_seed[seed_global_idx] = delta
            last_neighbor_by_seed[seed_global_idx] = z_idx

        if not muestras_sinteticas:
            for i in range(len(X_min)):
                self._log_muestra(
                    i, X, X_min, y, idxs_min_global,
                    comb, riesgo, densidades,
                    entropias, proporciones_min,
                    pureza_mask, densidad_mask,
                    umb_ent, None if self.percentil_densidad is None else float(self._meta["umbral_densidad"]),
                    vecinos_all_global, vecinos_min_global,
                    vecinos_validos_counts, dist_thr_por_muestra,
                    gen_from_counts, last_delta_by_seed, last_neighbor_by_seed
                )
            self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            return X.copy(), y.copy()

        X_sint = np.asarray(muestras_sinteticas, dtype=float)
        y_sint = np.ones(len(X_sint), dtype=int)

        X_resampled = np.vstack([X, X_sint])
        y_resampled = np.hstack([y, y_sint])

        # Registrar por muestra (con contadores completos)
        for i in range(len(X_min)):
            self._log_muestra(
                i, X, X_min, y, idxs_min_global,
                comb, riesgo, densidades,
                entropias, proporciones_min,
                pureza_mask, densidad_mask,
                umb_ent, None if self.percentil_densidad is None else float(self._meta["umbral_densidad"]),
                vecinos_all_global, vecinos_min_global,
                vecinos_validos_counts, dist_thr_por_muestra,
                gen_from_counts, last_delta_by_seed, last_neighbor_by_seed
            )

        self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        return X_resampled, y_resampled

    # --------------------- Multiclase ---------------------

    def fit_resample_multiclass(self, X, y):
        """
        Extiende a multiclase sobremuestreando cada clase contra la mayor,
        con factor_equilibrio. Aplica topes y concatena logs POR MUESTRA
        del run binario de cada clase (etiquetando 'clase_objetivo' y
        reordenándolo inmediatamente después de 'idx_global').
        """
        X = np.asarray(X)
        y = np.asarray(y)

        clases = np.unique(y)
        X_res = X.copy()
        y_res = y.copy()

        total_original = len(y)
        conteo_original = Counter(y)
        max_count = max(conteo_original.values())

        # Metadatos del experimento
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
                    max_total_multiplier=None,            # no se usa en el run binario aislado
                    max_sinteticas_por_clase=None,        # idem
                    guardar_distancias=self.guardar_distancias
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

                # Copiar LOG POR MUESTRA agregando clase_objetivo y reordenando
                for rec in sampler_tmp.logs_por_muestra:
                    rec_copia = dict(rec)
                    rec_copia["clase_objetivo"] = clase  # completar

                    # Reordenar: colocar clase_objetivo inmediatamente después de idx_global
                    keys = list(rec_copia.keys())
                    if "idx_global" in keys and "clase_objetivo" in keys:
                        k_cls = keys.pop(keys.index("clase_objetivo"))
                        keys.insert(keys.index("idx_global") + 1, k_cls)
                        rec_copia = {k: rec_copia[k] for k in keys}

                    self.logs_por_muestra.append(rec_copia)

                # Diagnóstico de motivo
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
                # 🔧 Fix de typo: "sobremuestreada"
                motivo = "sin_faltante(actual>=objetivo)" if estado != "sobremuestreada" else "tope=0"

            # Log POR CLASE (resumen)
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
                # Diagnóstico del filtro binario
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
