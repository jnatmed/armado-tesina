# pcs_smote.py

""" ----- Glosario ----- 
* fallback: mecanismo de respaldo o alternativa que se utiliza cuando el método principal falla o no está disponible. 
* ridge: técnica de regularización que añade una penalización a la magnitud de los coeficientes en modelos de regresión para prevenir el sobreajuste. 
* pseudo-inversa: generalización de la matriz inversa que puede aplicarse a matrices no cuadradas o singulares. 
* semilla: muestra de un dataset original que se usa como punto de partida para generar nuevas muestras sintéticas. 
* epsilom: valor muy pequeño usado para evitar divisiones por cero o inestabilidades numéricas. 
* varianza: medida de la dispersión de un conjunto de datos respecto a su media. Ejemplo: en una distribución normal, la varianza indica qué tan dispersos están los datos alrededor de la media. 
* covarianza: medida de cómo dos variables cambian juntas. Si ambas aumentan o disminuyen simultáneamente, la covarianza es positiva; si una aumenta mientras la otra disminuye, es negativa.
* traza: suma de los elementos en la diagonal principal de una matriz cuadrada.
""" 

from sklearn.neighbors import NearestNeighbors # para búsqueda de k vecinos más cercanos 
from sklearn.utils import check_random_state # para manejo de semilla y reproductibilidad 
from scipy.stats import entropy # para cálculo de entropía 
from collections import Counter, defaultdict # para conteos y diccionarios con valores por defecto 
import numpy as np # para cálculos numéricos 
import pandas as pd # para manejo de dataframes y exportación CSV 
import json # para exportación JSON 
import time # para medición de tiempos 
from sklearn.utils import check_random_state # para manejo de semilla y reproductibilidad

from Utils import Utils  # hereda utilidades comunes (reset/export/_dist/_log_muestra/etc.)

""" 
----- PC-SMOTE ----- 
Tecnica de sobremuestreo para datasets desbalanceados. La misma incorpora criterios de pureza y densidad y permite configurar el umbral de entropia y densidad. 

Se basa en la idea de que en datasets desbalanceados, las clases minoritarias son las que tienen menos ejemplos y, por lo tanto, tienen menor densidad.
"""

class PCSMOTE(Utils):
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

    def __init__(self, # se refiere al objeto mismo
                 k_neighbors=5,  # cantidad de vecinos a considerar
                 random_state=None, # semilla para reproductibilidad
                 radio_densidad=1.0, # radio para cálculo de densidad por intersección de esferas
                 percentil_dist=75, # percentil para umbral de distancia (Mahalanobis local)
                 percentil_entropia=None,  # percentil para umbral de entropía (si criterio_pureza='entropia')
                 percentil_densidad=None, # percentil para umbral de densidad (si se usa)
                 criterio_pureza='entropia', # 'entropia' o 'proporcion'
                 modo_espacial='2d', # '2d' o '3d' (afecta cálculo de distancias) --> posiblemente quede deprecado, ya que ahora el calculo pasa a ser por Mahalanobis
                 factor_equilibrio=0.8, # factor de equilibrio (0.0 … 1.0) para determinar cuántas sintéticas generar
                 verbose=True, # si es True, imprime mensajes de progreso y advertencias
                 max_total_multiplier=None, # tope global de crecimiento (multiplicador sobre la clase mayoritaria)
                 max_sinteticas_por_clase=None, # tope de sintéticas por clase (int)
                 guardar_distancias=True):  # ← por defecto True
        # Hiperparámetros
        self.k = int(k_neighbors) # cantodad de k vecinos mas cercanos a examinar
        self._seed_init = random_state # semilla fijada para reproductividad y mantener consistencias entre corridas
        self.random_state = check_random_state(random_state) # objeto RandomState para generación de aleatorios
        self.radio_densidad = float(radio_densidad) # radio para cálculo de densidad por intersección de esferas
        self.percentil_dist = float(percentil_dist) # percentil para umbral de distancia (Mahalanobis local)
        self.percentil_entropia = None if percentil_entropia is None else float(percentil_entropia) # percentil para umbral de entropía (si criterio_pureza='entropia')
        self.percentil_densidad = None if percentil_densidad is None else float(percentil_densidad) # percentil para umbral de densidad (si se usa)
        self.criterio_pureza = str(criterio_pureza) # 'entropia' o 'proporcion'
        self.modo_espacial = str(modo_espacial).lower()  # '2d' o '3d' 
        self.factor_equilibrio = float(factor_equilibrio) # factor de equilibrio (0.0 … 1.0) para determinar cuántas sintéticas generar
        self.verbose = bool(verbose) # si es True, imprime mensajes de progreso y advertencias
        self.guardar_distancias = bool(guardar_distancias) # si es True, guarda distancias a vecinos en log por muestra

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

        self._S_inv_by_i = None  # cache por semilla de la pseudo-inversa local        
        # ↓↓↓ Diagnósticos adicionales (no alteran comentarios previos)
        self._diag_maha = None            # diagnósticos por semilla (Mahalanobis)
        self._diag_umbral_maha_global = None  # resumen del umbral global
        self._diag_densidad = None        # contadores de densidad

    def _loggable_random_state(self):
        if isinstance(self._seed_init, (int, np.integer)):
            return int(self._seed_init)
        if self._seed_init is None:
            return None
        return str(self._seed_init)

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

        # preparar cache para cada semilla
        self._S_inv_by_i = [None] * n_min
        # preparar diagnósticos por semilla
        self._diag_maha = [None] * n_min

        """
        En cada iteración se toma la muestra minoritaria x_i, se recuperan sus k vecinos
        minoritarios y se organizan en una matriz; x_i se convierte a fila para operar
        vectorizadamente; se estima la covarianza local S de esos vecinos y se regulariza
        sumando λI para estabilizar la inversión; se calcula una pseudo-inversa estable
        de S_reg y, con ella, se obtienen las distancias de Mahalanobis de cada vecino
        a x_i, que se agregan a un acumulador global. Si ocurre algún problema numérico,
        se usa un fallback(*) a distancias euclídeas y también se acumulan. Al finalizar
        todas las iteraciones, ese conjunto de distancias servirá para fijar un umbral
        global por percentil que defina la “cercanía” en la métrica.
        """
        for i in range(n_min):                          # Itera por cada semilla minoritaria xi
            idx_nbrs = vecinos_min_local[i]             # Índices locales de los k vecinos minoritarios de xi
            if len(idx_nbrs) == 0:                      # Si una semilla (xi) no tiene vecinos, salta
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
                - trace(S)/p = promedio de varianzas (la “escala” típica). El numerador es la suma de varianzas, el denominador es la cantidad de features o dimensiones, lo que nos da como resultado el valor promedio de varianza por feature/ dimension.
                - max(1, n_feat) es un guardarraíl para no dividir por 0 en casos patológicos.
                En la práctica n_feat ≥ 1, pero este max asegura que si por error llegara n_feat = 0, dividas por 1 y no reviente
                """
                lam = 1e-3 * (np.trace(S) / max(1, n_feat)) # λ (lambda) proporcional a la escala de S (estable numéricamente)

                """
                - S: Es la matriz de covarianza estimada en el vecindario de Dimensión p x p (siendo p = n_feat).
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
                S_inv = np.linalg.pinv(S_reg, rcond=1e-8)  # Pseudo-inversa robusta ante S mal condicionada

                # cachear para usar la MISMA métrica en densidad
                self._S_inv_by_i[i] = S_inv

                # ---- Distancias de Mahalanobis de cada vecino respecto a xi ----
                diffs = nbrs - xi                       # Diferencias (k, n_feat)
                # d_M(xj, xi) = sqrt( (xj-xi)^T S_inv (xj-xi) ) para cada vecino
                d_maha = np.sqrt(np.einsum('ij,jk,ik->i', diffs, S_inv, diffs))  # Vector de k distancias

                if d_maha.size:                         # Si hay distancias válidas
                    todas.append(d_maha)                # Acumula para el percentil global
                    # diagnósticos por semilla
                    try:
                        singvals = np.linalg.svd(S_reg, compute_uv=False)
                        cond = float((singvals.max() / singvals.min()) if singvals.min() > 0 else np.inf)
                        rank = int(np.sum(singvals > 1e-12))
                    except Exception:
                        cond, rank = None, None
                    self._diag_maha[i] = {
                        "ok": True,
                        "fallback": False,
                        "n_vecinos": int(nbrs.shape[0]),
                        "traceS": float(np.trace(S)),
                        "lam": float(lam),
                        "rank_Sreg": rank,
                        "cond_Sreg": cond,
                        "d_maha_min": float(np.min(d_maha)),
                        "d_maha_med": float(np.median(d_maha)),
                        "d_maha_max": float(np.max(d_maha)),
                        "d_maha_mean": float(np.mean(d_maha)),
                        # "d_maha_list": d_maha.tolist(),  # ← opcional (puede ser grande)
                    }

            except Exception as e:
                self._S_inv_by_i[i] = None  # sin métrica local, caemos a euclídea luego
                # Fallback local: si falla Mahalanobis (cov/inversa/einsum), usar Euclídea
                try:
                    d_euc = np.linalg.norm(nbrs - xi, axis=1)
                    if d_euc.size:
                        todas.append(d_euc)
                    if getattr(self, "verbose", False):
                        print(f"[PCSMOTE][warn] Mahalanobis falló en i={i}. Fallback a euclídea. Detalle: {e}")
                    self._diag_maha[i] = {
                        "ok": False,
                        "fallback": True,
                        "n_vecinos": int(nbrs.shape[0]),
                        "traceS": None,
                        "lam": None,
                        "rank_Sreg": None,
                        "cond_Sreg": None,
                        "d_maha_min": None,
                        "d_maha_med": None,
                        "d_maha_max": None,
                        "d_maha_mean": None,
                        "d_euc_min": float(np.min(d_euc)) if d_euc.size else None,
                        "d_euc_med": float(np.median(d_euc)) if d_euc.size else None,
                        "d_euc_max": float(np.max(d_euc)) if d_euc.size else None,
                        "d_euc_mean": float(np.mean(d_euc)) if d_euc.size else None,
                    }
                except Exception as e2:
                    if getattr(self, "verbose", False):
                        print(f"[PCSMOTE][warn] Euclídea también falló en i={i}. Se omite muestra. Detalle: {e2}")
                    self._diag_maha[i] = {"ok": False, "fallback": True}
                    continue

        if len(todas) == 0:                             # Si no se pudo calcular nada (sin vecinos en todo el set)
            self.umbral_distancia = 0.0                 # Umbral neutro (no contará “cercanos”)
            # resumen global
            self._diag_umbral_maha_global = {
                "percentil": float(percentil),
                "umbral_global": float(self.umbral_distancia)
            }
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

        # resumen global
        self._diag_umbral_maha_global = {
            "percentil": float(percentil),
            "umbral_global": float(self.umbral_distancia)
        }

    def calcular_densidad_interseccion(self, X_min, vecinos_local):
        """
        Densidad por intersección de esferas de radio 'radio', usando índices locales de X_min.
        """
        densidades = []
        umbral_distancia = self.getUmbralDistancia()  # umbral global (float)        

        # contadores opcionales para diagnóstico
        self._diag_densidad = {"semillas_con_hits": 0, "total_hits": 0}

        for i, xi in enumerate(X_min):
            intersecciones = 0
            for j in vecinos_local[i]:
                xj = X_min[j]

                S_inv = None
                if isinstance(self._S_inv_by_i, list) and i < len(self._S_inv_by_i):
                    S_inv = self._S_inv_by_i[i]

                if S_inv is not None:
                    # d_M(xj, xi) = sqrt( (xj-xi)^T S_inv (xj-xi) )
                    diff = (xj - xi).reshape(1, -1)
                    d = float(np.sqrt(np.einsum('ij,jk,ik->i', diff, S_inv, diff)))
                else:
                    # fallback coherente si no hubo S_inv para esta semilla
                    d = float(np.linalg.norm(xi - xj))

                if d <= umbral_distancia:
                    intersecciones += 1
                    
            if intersecciones > 0:
                self._diag_densidad["semillas_con_hits"] += 1
                self._diag_densidad["total_hits"] += intersecciones

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

        # Calcula y guarda el umbral global de distancia (Mahalanobis local)
        self.distancia_x_mahalanobis(
            X_min=X_min,
            vecinos_min_local=vecinos_min_local,
            percentil=self.percentil_dist  # o un percentil específico si preferís
        )

        densidades = self.calcular_densidad_interseccion(X_min, vecinos_min_local)

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

                # (dentro de fit_resample_multiclass, inmediatamente después
                #  de copiar logs_por_muestra desde sampler_tmp)

                # --- ACUMULAR DIAGNÓSTICOS MAHALANOBIS EN EL OBJETO PRINCIPAL ---
                if getattr(sampler_tmp, "_diag_maha", None):
                    # inicializar contenedor si hace falta
                    if getattr(self, "_diag_maha", None) is None or isinstance(self._diag_maha, list) and len(self._diag_maha) == 0:
                        self._diag_maha = []

                    for idx_local, d in enumerate(sampler_tmp._diag_maha):
                        if d is None:
                            continue
                        d_copy = dict(d)
                        # conservar clase objetivo y el índice local dentro de esa corrida
                        d_copy["clase_objetivo"] = clase
                        d_copy["idx_local_run"] = int(idx_local)
                        self._diag_maha.append(d_copy)

                # opcional: también conservar un resumen del umbral global de esa corrida
                if getattr(sampler_tmp, "_diag_umbral_maha_global", None):
                    if not hasattr(self, "_diag_umbral_maha_global_list"):
                        self._diag_umbral_maha_global_list = []
                    g = dict(sampler_tmp._diag_umbral_maha_global)
                    g["clase_objetivo"] = clase
                    self._diag_umbral_maha_global_list.append(g)


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
