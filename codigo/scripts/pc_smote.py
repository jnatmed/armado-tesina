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

from sklearn.neighbors import NearestNeighbors  # para búsqueda de k vecinos más cercanos 
from sklearn.utils import check_random_state     # para manejo de semilla y reproductibilidad 
from scipy.stats import entropy                  # para cálculo de entropía 
from collections import Counter, defaultdict     # para conteos y diccionarios con valores por defecto 
import numpy as np                               # para cálculos numéricos 
import pandas as pd                              # para manejo de dataframes y exportación CSV 
import json                                      # para exportación JSON 
import time                                      # para medición de tiempos 

from Utils import Utils  # hereda utilidades comunes (reset/export/_dist/_log_muestra/etc.)
# pcs_smote.py
from cache import PCSMOTECache
from gestor_cache import PCSMOTEGestorCache

""" 
----- PC-SMOTE ----- 
Técnica de sobremuestreo para datasets desbalanceados. Incorpora criterios de pureza y densidad
y permite configurar los umbrales correspondientes.

En ALTA DIMENSION, utilizamos **Local Scaling Distance (LSD)**:
    d_LS(x_i, x_j) = ||x_i - x_j||_2 / sqrt(sigma_i * sigma_j)
donde sigma_i es la distancia euclídea de x_i a su k_sigma-ésimo vecino (self-scaling).
Esto evita inversión de covarianzas, mantiene contraste local y mejora estabilidad.
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
    umbral_distancia = None  # umbral global (float)

    # ---- Configuración de métrica de vecindario (solo LSD) ----
    metric_vecindario: str = "lsd"

    def __init__(self,
                 k_neighbors=7,
                 random_state=None,
                 radio_densidad=1.0,
                 percentil_dist=75,              # percentil para umbral de distancia (LSD)
                 percentil_entropia=None,        # percentil para umbral de entropía (si criterio_pureza='entropia')
                 percentil_densidad=None,        # percentil para umbral de densidad (si se usa)
                 criterio_pureza='entropia',     # 'entropia' o 'proporcion'
                 modo_espacial='2d',             # mantenido por retrocompatibilidad
                 factor_equilibrio=0.8,
                 verbose=True,
                 max_total_multiplier=None,
                 max_sinteticas_por_clase=None,
                 guardar_distancias=True,
                 metric_vecindario="lsd"):  
        # Hiperparámetros
        self.k = int(k_neighbors) # cantidad de vecinos más cercanos de la misma clase
        self._seed_init = random_state # semilla inicial
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
        self.metric_vecindario = metric_vecindario

        # ---- Gestión de caché ----
        # Sirve para traer o guardar parametros que 
        # se vuelven a utilizar entre corridas
        # que no vale la pena o mejor dicho es costoso volver
        # a calcular
        self.cache = PCSMOTECache() 

        # Crear el gestor intermedio usando los parámetros actuales
        self.gestor_cache = PCSMOTEGestorCache(
            cache=self.cache,
            k=self.k,
            metrica_vecindario=self.metric_vecindario,
            percentil_dist=self.percentil_dist,
        )

        # ---- LSD (Local Scaling Distance) ----
        self._sigma_X = None            # sigma por punto en X (distancia al k-ésimo vecino)
        self._sigma_Xmin = None         # sigma por punto en X_min
        self._umbral_lsd_by_i = None    # umbral LOCAL LSD por semilla

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

        # Diagnósticos de densidad
        self._diag_densidad = None

    # -------------------------------
    #          LSD helpers
    # -------------------------------
    def _compute_sigmas(self, X: np.ndarray, k_sigma: int) -> np.ndarray:
        """
        Devuelve sigma[i] = distancia euclídea desde X[i] a su k_sigma-ésimo vecino.
        Si k_sigma < 1 o hay pocos puntos, usa fallback estable.
        """
        X = np.asarray(X)
        n = len(X)
        if n <= 1:
            return np.ones(n, dtype=float)

        k_sigma = max(1, min(k_sigma, n - 1))  # evita pedir más vecinos que puntos-1
        nn = NearestNeighbors(n_neighbors=k_sigma + 1).fit(X)
        dists, _ = nn.kneighbors(X, return_distance=True)
        # dists[:, 0] = 0 (self), tomamos el k_sigma-ésimo
        sigmas = dists[:, k_sigma].astype(float)
        # evita ceros exactos
        sigmas[sigmas <= 1e-12] = 1e-12
        return sigmas

    def _dists_lsd_seed(self, xi: np.ndarray, Xref: np.ndarray,
                        sigma_i: float, sigmas_ref: np.ndarray) -> np.ndarray:
        """
        d_LS(xi, xj) = ||xi - xj||_2 / sqrt(sigma_i * sigma_j)
        """
        diffs = Xref - xi.reshape(1, -1)
        euc = np.linalg.norm(diffs, axis=1)
        denom = np.sqrt(sigma_i * sigmas_ref)
        denom[denom <= 1e-18] = 1e-18
        return euc / denom

    def _loggable_random_state(self):
        if isinstance(self._seed_init, (int, np.integer)):
            return int(self._seed_init)
        if self._seed_init is None:
            return None
        return str(self._seed_init)

    def getUmbralDistancia(self):
        return 0.0 if self.umbral_distancia is None else float(self.umbral_distancia)

    # ------------------------------------------
    #   Umbral global/local por LSD
    # ------------------------------------------
    def calcularUmbralDensidades(self, X_min, vecinos_min_local, percentil=75.0, k_sigma=None):
        """
        Calcula UMBRAL GLOBAL usando Local Scaling Distance (LSD) y también
        fija un umbral LOCAL por semilla (percentil sobre sus k vecinos minoritarios).
        Guarda:
          - self.umbral_distancia (global)
          - self._umbral_lsd_by_i (local por semilla)
          - self._sigma_Xmin (para densidad en minoritaria)
        """
        X_min = np.asarray(X_min)
        n_min = len(X_min)
        if n_min == 0:
            self.umbral_distancia = 0.0
            self._umbral_lsd_by_i = np.array([], dtype=float)
            return

        # k para el radio local (self-scaling). Por defecto, se usa self.k
        k_sigma = int(self.k if k_sigma is None else k_sigma)

        todas = []
        self._umbral_lsd_by_i = np.full(n_min, np.nan, dtype=float)

        """
        Recorro cada semilla i en X_min, calculo distancias LSD a sus vecinos
        minoritarios y fijo umbral local por semilla.
        """
        for i in range(n_min):
            idx_nbrs = vecinos_min_local[i]
            if len(idx_nbrs) == 0:
                continue # si no tiene vecinos, no hay umbral local pasar al siguiente
            
            # idx_nbrs son índices de los vecinos LOCALES en X_min
            nbrs = X_min[idx_nbrs] # vecinos minoritarios de la semilla i
            sigma_i = float(self._sigma_Xmin[i])
            sigmas_vec = self._sigma_Xmin[idx_nbrs]

            """
            Distancias LSD desde la semilla i a sus vecinos minoritarios.
            tengo que enviarle los 2 sigmas (de la semilla y de los vecinos)
            y dentro de _dists_lsd_seed() se hace el cálculo.
            """
            d_lsd = self._dists_lsd_seed(X_min[i], nbrs, sigma_i, sigmas_vec)

            # Si hay distancias entonces las almaceno
            if d_lsd.size:
                todas.append(d_lsd)
                # Paso importante, guardo umbral LOCAL por semilla
                self._umbral_lsd_by_i[i] = float(np.percentile(d_lsd, percentil))

        if len(todas) == 0:
            self.umbral_distancia = 0.0
        else:
            try:
                todas = np.concatenate(todas)
                self.umbral_distancia = float(np.percentile(todas, percentil))
            except Exception:
                self.umbral_distancia = 0.0

    # ------------------------------------------
    #   Densidad por intersección (con LSD)
    # ------------------------------------------
    def calcular_densidad_interseccion(self, X_min, vecinos_local):
        """
        Densidad por intersección de esferas usando LSD y el umbral local u_i (o global si no hay local).
        """
        densidades = []

        # contadores opcionales para diagnóstico
        self._diag_densidad = {"semillas_con_hits": 0, "total_hits": 0}

        for i, xi in enumerate(X_min):

            intersecciones = 0

            # umbral LOCAL 
            u_i = self._umbral_lsd_by_i[i]

            # LSD: necesitamos sigma_i y sigma_j (ambos en X_min)
            if self._sigma_Xmin is None or len(self._sigma_Xmin) != len(X_min):
                sigma_i = 1.0
                sigmas_local = np.ones(len(X_min), dtype=float)
            else:
                sigma_i = float(self._sigma_Xmin[i])
                sigmas_local = self._sigma_Xmin

            for j in vecinos_local[i]:
                xj = X_min[j]
                d = float(self._dists_lsd_seed(
                    xi, xj.reshape(1, -1),
                    sigma_i,
                    np.array([sigmas_local[j]], dtype=float)
                )[0])
                if d <= u_i:
                    intersecciones += 1

            if intersecciones > 0:
                self._diag_densidad["semillas_con_hits"] += 1
                self._diag_densidad["total_hits"] += intersecciones

            # guarda en densidades la proporción de intersecciones por semilla
            # (intersecciones / cantidad de vecinos minoritarios)
            # max(1, ...) para evitar división por cero
            densidades.append(intersecciones / max(1, len(vecinos_local[i])))


        # en el array devuelvo porcentaje de densidades por semilla
        return np.array(densidades, dtype=float)

    def calcular_entropia(self, vecinos_all_global, y):
        """Entropía de clases en el vecindario (base 2)."""
        entropias = []
        for idxs in vecinos_all_global:
            clases, counts = np.unique(y[idxs], return_counts=True)
            p = counts / counts.sum()
            entropias.append(float(entropy(p, base=2)))
        return np.array(entropias, dtype=float)

    # ------------------------------------------
    #                FIT / RESAMPLE
    # ------------------------------------------
    def fit_resample(self, X, y, max_sinteticas=None):
        """
        Ejecuta el proceso completo de sobremuestreo binario (y ∈ {0,1}) 
        basado en LSD, riesgo, densidad y pureza.

        Devuelve:
            (X_resampled, y_resampled)
        
        Efectos:
            - Calcula distancias LSD globales/locales.
            - Selecciona semillas candidatas según riesgo, densidad y pureza.
            - Genera muestras sintéticas por interpolación adaptativa.
            - Registra métricas agregadas en self._meta y logs por muestra.
        """
        t0 = time.perf_counter()
        X = np.asarray(X)
        y = np.asarray(y)

        # ------------------------------------------------------------------
        # (1) Inicialización de metadatos y separación de clases
        # ------------------------------------------------------------------
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

        # Separacion de clases
        idxs_min_global = np.where(y == 1)[0]
        idxs_maj_global = np.where(y == 0)[0]
        X_min = X[idxs_min_global]
        X_maj = X[idxs_maj_global]

        # Si no hay minoritaria o no hay suficientes para k+1 vecinos, retorno sin cambios
        if len(X_min) < self.k + 1:
            self._meta.update({
                "n_candidatas": int(len(X_min)),
                "n_filtradas": 0,
                "elapsed_ms": (time.perf_counter() - t0) * 1000
            })
            # Retorno sin cambios
            return X.copy(), y.copy()

        """
        ------------------------------------------------------------------
        (2) Distancias LSD y vecindarios (con caché y gestor_cache)
        ------------------------------------------------------------------
        Si es la primera ejecucion con estos parámetros, calcula y guarda.
        Si ya se corrió antes, trae los resultados guardados y evita cálculos.
        Se le envia la propia instancia de pcsmote, para utilizarla 
        como adaptador, para que pueda llamar a los métodos auxiliares y 
        obtener los resultados por primera vez o traerlos de la caché.
        Previamente en el construct de pcsmote ya se inicializo gestor_cache con
        el objeto cache para que pueda acceder a los datos persistidos. 
        """
        (vecinos_all_global,
        vecinos_min_local,
        vecinos_min_global,
        self._sigma_X,
        self._sigma_Xmin) = self.gestor_cache.obtener(
            X=X,
            y=y,
            nombre_dataset=self.nombre_dataset,
            adaptador=self
        )

        # ------------------------------------------------------------------
        # (3) Cálculo de métricas locales: riesgo, densidad y pureza
        # ------------------------------------------------------------------
        # (3-a) Riesgo local: proporción de vecinos mayoritarios
        riesgo = np.array([
            np.sum(y[idxs] == 0) / self.k for idxs in vecinos_all_global
        ], dtype=float)

        # (3-b) Densidad local por intersección de esferas (LSD)
        # ¿ Que tan denso es el vecindario de xi ?
        densidades = self.calcular_densidad_interseccion(X_min, vecinos_min_local)
        # Resultado => ejemplo: densidades = [0.05, 0.20, 0.35, 0.50, 0.80, 0.95]
        # porcentajes de intersección por semilla   

        # (3-c) Pureza del vecindario
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
                np.sum(y[idxs] == 1) / self.k for idxs in vecinos_all_global
            ], dtype=float)
            pureza_mask = (proporciones_min >= 0.4) & (proporciones_min <= 0.6)
        else:
            raise ValueError(f"Criterio de pureza no reconocido: {self.criterio_pureza}")

        # ------------------------------------------------------------------
        # (4) Filtrado de semillas candidatas según densidad y pureza
        # ------------------------------------------------------------------
        """
            densidades = [0.05, 0.20, 0.35, 0.50, 0.80, 0.95]
            percentil_densidad = 75
            umb_den = np.percentile(densidades, 75)  # ≈ 0.725        
        """
        if self.percentil_densidad is not None:
            umb_den = float(np.percentile(densidades, self.percentil_densidad))

            # En densidad_mask quedaran los indices de las semillas  que
            # superen el umbral de densidad
            densidad_mask = densidades >= umb_den
            self._meta["umbral_densidad"] = umb_den
        else:
            umb_den = None
            densidad_mask = densidades > 0.0

        # Hace la interseccion de los arrays de pureza y densidad
        # quedan los indices de las semillas que cumplen con ambos criterios
        comb = pureza_mask & densidad_mask

        # luego comb se pasa por np.where que lo que hace es
        # devolver los indices de los elementos que cumplen con el criterio
        # Sera candidatos a generar sinteticos
        filtered_indices_local = np.where(comb)[0]

        # indices globales de las semillas filtradas
        # sirve para loggear correctamente
        filtered_indices_global = idxs_min_global[filtered_indices_local]

        self._meta.update({
            "n_candidatas": int(len(X_min)),
            "n_filtradas": int(np.sum(comb)),
            "riesgo_medio": float(np.mean(riesgo[comb])) if np.any(comb) else None,
            "riesgo_std": float(np.std(riesgo[comb])) if np.any(comb) else None,
            "densidad_media": float(np.mean(densidades)) if densidades.size else None
        })

        # ------------------------------------------------------------------
        # (5) Diagnóstico: vecinos válidos y umbral por percentil LSD
        # ------------------------------------------------------------------
        vecinos_validos_counts = np.zeros(len(X_min), dtype=int)
        dist_thr_por_muestra = np.full(len(X_min), np.nan)

        for i in range(len(X_min)):
            idxs_vec_all = vecinos_all_global[i]
            xi = X_min[i]
            sigma_i = float(self._sigma_Xmin[i]) if (self._sigma_Xmin is not None and i < len(self._sigma_Xmin)) else 1.0
            sigmas_ref = self._sigma_X[idxs_vec_all] if self._sigma_X is not None else np.ones(len(idxs_vec_all))
            dists = self._dists_lsd_seed(xi, X[idxs_vec_all], sigma_i, sigmas_ref)
            thr = np.percentile(dists, self.percentil_dist)
            dist_thr_por_muestra[i] = float(thr)
            vecinos_validos_counts[i] = int(np.sum(dists <= thr))

        self._meta["vecinos_validos_promedio"] = float(np.mean(vecinos_validos_counts))

        gen_from_counts = defaultdict(int)
        last_delta_by_seed = {}
        last_neighbor_by_seed = {}

        # ------------------------------------------------------------------
        # (6) Verificaciones previas a la generación
        # ------------------------------------------------------------------
        # Si no hay semillas candidatas, o no hay suficientes para k+1 vecinos,
        # se regresa el conjunto original
        if len(filtered_indices_local) < self.k + 1:
            self._registrar_logs_sin_sinteticas(
                X, y, X_min, idxs_min_global,
                comb, riesgo, densidades, entropias, proporciones_min,
                pureza_mask, densidad_mask,
                umb_ent, umb_den,
                vecinos_all_global, vecinos_min_global,
                vecinos_validos_counts, dist_thr_por_muestra
            )
            self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000

            # retorno sin cambios
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

        # ------------------------------------------------------------------
        # (7) Generación de muestras sintéticas por interpolación
        # ------------------------------------------------------------------
        X_min_filtrado = X_min[filtered_indices_local]
        vecinos_all_filtrado = vecinos_all_global[filtered_indices_local]
        muestras_sinteticas = []

        for _ in range(n_sint):
            idx_local_filt = self.random_state.randint(len(X_min_filtrado))
            xi = X_min_filtrado[idx_local_filt]
            r_i = riesgo[filtered_indices_local][idx_local_filt]

            i_local_orig = int(filtered_indices_local[idx_local_filt])
            idxs_vec_all = vecinos_all_filtrado[idx_local_filt]
            sigma_i = float(self._sigma_Xmin[i_local_orig]) if (self._sigma_Xmin is not None and i_local_orig < len(self._sigma_Xmin)) else 1.0
            sigmas_ref = self._sigma_X[idxs_vec_all] if self._sigma_X is not None else np.ones(len(idxs_vec_all))
            dists = self._dists_lsd_seed(xi, X[idxs_vec_all], sigma_i, sigmas_ref)

            thr = np.percentile(dists, self.percentil_dist)
            vecinos_validos = idxs_vec_all[dists <= thr]
            if len(vecinos_validos) == 0:
                continue

            z_idx = int(self.random_state.choice(vecinos_validos))
            xz = X[z_idx]

            # Delta adaptativo según riesgo local
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

        # ------------------------------------------------------------------
        # (8) Concatenación, logging y retorno final
        # ------------------------------------------------------------------
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
    #           Multiclase (one-vs-max)
    # ------------------------------------------
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
