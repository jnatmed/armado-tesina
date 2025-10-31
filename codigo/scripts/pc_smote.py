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

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.metrics import DistanceMetric
from scipy.stats import entropy
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import time

import traceback
from Utils import Utils  # clase utilitaria


class PCSMOTE(Utils):
    """
    PC-SMOTE (refactor):
      - Métricas de distancia de scikit-learn (DistanceMetric/NearestNeighbors).
      - Densidad por intersección en subespacio minoritario.
      - Filtro por pureza (entropía o proporción), riesgo local y densidad.
      - Logs por clase y por muestra, y traza detallada de decisiones.

    Convenciones:
      • Binario en fit_resample: y ∈ {0,1}, 1 = clase minoritaria.
      • Multiclase vía one-vs-max en fit_resample_multiclass.
    """

    DELTA_RANGE_INTERMEDIO = (0.4, 0.6)

    # al pasar * como primer argumento, fuerzo a que todos los argumentos sean keywords
    # el constructor solo va a aceptar que pcsmote se instancie con keywords
    # ej: pcsmote = PCSMOTE(k_neighbors=5, random_state=42, percentil_dist=75, ...)
    # ejemplos no validos son pcsmote = PCSMOTE(5, 42, 75, ...) => esto daria error. 
    def __init__(self, *,
                 k_neighbors=7,  #1 
                 random_state=None, #2 random_state, osea semilla inicial
                 # -- Hiperparámetros de PC-SMOTE --
                 percentil_dist=75,  #3 percentil de distancia para densidad y vecinos válidos
                 percentil_entropia=None, #4 percentil de entropía para filtro de pureza
                 percentil_densidad=None, #5 percentil de densidad para filtro
                 criterio_pureza='entropia',   # 'entropia' | 'proporcion' #6 criterio de pureza
                 factor_equilibrio=1, # 7 factor de equilibrio en multiclase
                 verbose=True, #8 modo verbose, si quiero que se vean los prints, utiles para el debug
                 max_total_multiplier=None, #8 tope global de sintéticas (multiplicador del total original)
                 max_sinteticas_por_clase=None, #9 tope por clase
                 guardar_distancias=True, #10 si quiero guardar distancias en logs por muestra
                 metric='euclidean'): #11 métrica de distancia elegida
        # Hiperparámetros esenciales
        self.k = int(k_neighbors)
        self._seed_init = random_state # para logging
        # la funcion check_random_state maneja None, int, y RandomState
        # es decir, si le paso None crea un RandomState aleatorio
        # si le paso un int crea un RandomState con esa semilla
        self.random_state = check_random_state(random_state)

        # Hiperparámetros de PC-SMOTE
        
        self.percentil_dist = float(percentil_dist)
        self.percentil_entropia = None if percentil_entropia is None else float(percentil_entropia)
        self.percentil_densidad = None if percentil_densidad is None else float(percentil_densidad)
        self.criterio_pureza = str(criterio_pureza)

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
        self.logs_por_clase = []      # lista de dicts
        self.logs_por_muestra = []    # si tu Utils lo llena, quedan aquí
        self.meta_experimento = {}    # resumen global multiclase
        self._meta = {}               # resumen binario de la última corrida
        self.debug_trace = []         # traza fina de decisiones (eventos)

        # Nombre de dataset (si el flujo externo lo setea)
        self.nombre_dataset = getattr(self, "nombre_dataset", "unknown")

        # Diagnóstico opcional
        self._diag_densidad = None

    # ------------------------------- Densidad por intersección -------------------------------

    def calcular_densidad_interseccion(self, X_min, vecinos_local, dists_min_local):
        """
        Densidad por intersección entre semillas MINORITARIAS:
          - Para cada semilla i, u_i = percentil(self.percentil_dist) de distancias a k vecinos minoritarios.
          - Cuenta fracción de vecinos a distancia <= u_i (con DistanceMetric seleccionado).
        """
        X_min = np.asarray(X_min)
        n_min = len(X_min)
        if n_min == 0:
            return np.array([], dtype=float)

        densidades = np.zeros(n_min, dtype=float)
        self._diag_densidad = {"semillas_con_hits": 0, "total_hits": 0}

        # Para cada semilla
        for i in range(n_min):
            # Vecinos minoritarios
            # Obtener índices de vecinos minoritarios locales
            nbr_idx_local = vecinos_local[i]
            # Si no hay vecinos, densidad 0
            # Paso a la siguiente semilla minoritaria
            if len(nbr_idx_local) == 0:
                densidades[i] = 0.0
                continue

            # Obtener distancias a vecinos minoritarios    
            d_i = dists_min_local[i]  # (k,)
            # Calcular umbral u_i    
            u_i = float(np.percentile(d_i, self.percentil_dist))

            # reshape lo que hace es tomar la fila i y convertirla en una matriz 1xd
            xi = X_min[i].reshape(1, -1)
            # Obtengo los vecinos minoritarios de xi
            # tomando nbr_idx_local como índices sobre X_min
            xj = X_min[nbr_idx_local]  # (k, d)

            # Aca uso DistanceMetric
            # pairwise devuelve (1, k), que se refiere a las distancias de xi a cada xj
            dij = self._dist_metric.pairwise(xi, xj).ravel()  # (k,)

            # Si alguna de las distancias a cada vecino
            # es menor o igual al umbral entonces lo cuento como
            # interseccion
            intersecciones = int(np.sum(dij <= u_i))

            # Si hay intersecciones, actualizo diagnósticos
            if intersecciones > 0:
                # semillas_con_hits: cuenta cuántas semillas tienen al menos un vecino dentro del umbral.
                self._diag_densidad["semillas_con_hits"] += 1
                # total_hits: suma total de vecinos que están dentro del umbral para todas las semillas.
                # Osea es mas una estadistica global de distancias por debajo del umbral
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

    # -------------------------------------------- FIT / RESAMPLE (BINARIO) --------------------------------------------

    def fit_resample(self, X, y, max_sinteticas=None):
            """
            Oversampling binario (y ∈ {0,1}, 1 = minoritaria).
            Mantiene SIEMPRE todas las semillas originales; solo apendea sintéticas.
            
            Args:
                X (np.ndarray): Características del dataset.
                y (np.ndarray): Etiquetas del dataset (binarias, 1 es minoritaria).
                max_sinteticas (int, optional): Número máximo de muestras sintéticas a generar. 
                                                Si es None, intenta igualar el tamaño de la clase mayoritaria.

            Returns:
                tuple: (X_resampled, y_resampled), el dataset de entrada con las muestras sintéticas agregadas.
            """
            t0 = time.perf_counter()
            X = np.asarray(X)
            y = np.asarray(y)

            # Meta base con k_efectivo inicializado
            # Inicialización de un diccionario para almacenar metadatos del proceso de remuestreo.
            self._meta = {
                "umbral_densidad": None,
                "umbral_entropia": None,
                "riesgo_medio": None,
                "riesgo_std": None,
                "densidad_media": None,
                "vecinos_validos_promedio": None,
                "n_candidatas": None,
                "n_filtradas": None,
                "elapsed_ms": None,
                "k_efectivo": None,
            }

            # Separación
            # Identificación de los índices y separación de las clases minoritaria (1) y mayoritaria (0).
            idxs_min_global = np.where(y == 1)[0]
            idxs_maj_global = np.where(y == 0)[0]
            X_min = X[idxs_min_global]
            X_maj = X[idxs_maj_global]

            K = int(self.k)
            self._meta["k_efectivo"] = K  # k_efectivo, es el k que se usará en esta corrida

            # Condición mínima: k+1 para poder excluir self
            # Primera salida temprana: Insuficientes muestras minoritarias para calcular vecindades.
            # Si la clase minoritaria es menor a la cantidad de vecinos 
            # necesarios para calcular vecindades, no se puede calcular la densidad
            if len(X_min) < (K + 1):
                # Se registra la información de salida temprana en los metadatos.
                self._meta.update({
                    "n_candidatas": int(len(X_min)),
                    "n_filtradas": 0,
                    "vecinos_validos_promedio": 0,
                    "elapsed_ms": (time.perf_counter() - t0) * 1000,
                    "k_efectivo": K,
                })
                # No genera sintéticas: devuelve copia idéntica
                # RETURN 1: Retorna X e y sin modificaciones (copias) si la clase minoritaria es muy pequeña.
                # Razón: No hay suficientes muestras para establecer un vecindario de tamaño K+1, lo que impide 
                # calcular métricas basadas en k-NN (como la densidad o el riesgo).
                return X.copy(), y.copy()

            # Vecindarios
            # Cálculo de los k-vecinos más cercanos (k+1 para incluir la propia muestra)
            # 1. Vecindario Global (NN sobre todo X)
            # Lo necesito para calcular riesgo y pureza
            nn_global = NearestNeighbors(n_neighbors=K + 1, metric=self.metric).fit(X)
            d_all, i_all = nn_global.kneighbors(X_min, return_distance=True)  # incluye self en col 0
            d_all = d_all[:, 1:]  # (n_min, K) -> Distancias a los K vecinos más cercanos (excluyendo a sí misma).
            i_all = i_all[:, 1:]  # (n_min, K) índices sobre X -> Índices globales de los K vecinos.

            # 2. Vecindario Minoritario (NN sobre X_min)
            nn_min = NearestNeighbors(n_neighbors=K + 1, metric=self.metric).fit(X_min)

            # Obtencion de los vecinos de cada semilla minoritaria dentro del subespacio minoritario
            d_min, i_min_local = nn_min.kneighbors(X_min, return_distance=True)
            d_min = d_min[:, 1:]  # (n_min, K) -> Distancias a los K vecinos minoritarios más cercanos (excluyendo a sí misma).
            i_min_local = i_min_local[:, 1:]  # (n_min, K) índices locales sobre X_min -> Índices locales de los K vecinos minoritarios.

            vecinos_all_global = i_all
            vecinos_min_local = i_min_local

            # Conversión de índices locales minoritarios a índices globales.
            vecinos_min_global = np.array([idxs_min_global[row] for row in vecinos_min_local], dtype=object)

            # Métricas locales
            # 1. Cálculo del Riesgo (proporción de vecinos mayoritarios)
            # Riesgo: Proporción de muestras de la clase mayoritaria (y=0) entre los K vecinos globales. 
            # Un riesgo alto indica una muestra más "cerca" de la frontera o en la región mayoritaria.
            riesgo = np.array([np.sum(y[idxs] == 0) / K for idxs in vecinos_all_global], dtype=float)

            # 2. Cálculo de la Densidad
            # Densidad: Mide qué tan "compacta" está la muestra minoritaria con respecto a sus vecinos minoritarios.
            densidades = self.calcular_densidad_interseccion(
                # Envio las semillas minoritarias
                X_min=X_min,
                # Envio las distancias a los vecinos minoritarios
                vecinos_local=vecinos_min_local,
                dists_min_local=d_min
            )
        # k=7
        # 1/7
        # .25 
            pureza_mask = None
            umb_ent = None
            entropias = None
            proporciones_min = None

            # Aplicación del Criterio de Pureza
            if self.criterio_pureza == 'entropia':
                # Criterio: Se utiliza la entropía de la vecindad global para medir la "pureza".
                entropias = self.calcular_entropia(vecinos_all_global, y)
                # umb_ent: Umbral de entropía basado en el percentil configurado.
                umb_ent = float(np.percentile(entropias, self.percentil_entropia)) if self.percentil_entropia else None
                # pureza_mask: True para las muestras con entropía menor o igual al umbral (más puras o de frontera).
                pureza_mask = entropias <= (umb_ent if umb_ent is not None else 1.0)
                self._meta["umbral_entropia"] = umb_ent
            elif self.criterio_pureza == 'proporcion':
                # proporción de minoritarios en el vecindario global (granularidad de 1/K)
                proporciones_min = (y[vecinos_all_global] == 1).mean(axis=1).astype(float)
                # ventana de “frontera” adaptativa: al menos 1 vecino minoritario y 1 mayoritario
                eps = 1.0 / K
                pureza_mask = (proporciones_min >= eps) & (proporciones_min <= (1.0 - eps))
            else:
                # criterio desconocido -> no generamos sintéticas, devolvemos tal cual
                # Salida temprana: Criterio de pureza no reconocido.
                self._meta.update({
                    "n_candidatas": int(len(X_min)),
                    "n_filtradas": 0,
                    "vecinos_validos_promedio": 0,
                    "elapsed_ms": (time.perf_counter() - t0) * 1000,
                    "k_efectivo": K,
                })
                # RETURN 2: Retorna X e y sin modificaciones si el 'criterio_pureza' no es válido.
                # Razón: El algoritmo no puede determinar las semillas válidas sin un criterio de pureza/frontera.
                return X.copy(), y.copy()

            # Aplicación del Criterio de Densidad
            if self.percentil_densidad is not None:
                # Se establece un umbral de densidad basado en el percentil configurado.
                """
                Osea de las densidades calculadas, se toma el percentil configurado (ej 75)
                y se usa como umbral. Solo las muestras con densidad mayor o igual a ese umbral
                se consideran válidas para la generación sintética.
                """
                umb_den = float(np.percentile(densidades, self.percentil_densidad))
                # densidad_mask: True para las muestras con densidad mayor o igual al umbral (más densas).
                densidad_mask = densidades >= umb_den
                self._meta["umbral_densidad"] = umb_den
            else:
                umb_den = None
                # Si no se define percentil, simplemente se filtran las muestras con densidad > 0.
                densidad_mask = densidades > 0.0

            # Combinación de filtros (Pureza Y Densidad)
            # comb: Máscara final de las semillas minoritarias válidas para generar muestras sintéticas.
            comb = pureza_mask & densidad_mask
            filtered_indices_local = np.where(comb)[0]  # Índices locales sobre X_min
            filtered_indices_global = idxs_min_global[filtered_indices_local]  # Índices globales sobre X

            # Diagnóstico vecinos válidos (por percentil de distancias globales)
            # Cálculo de una máscara de "vecinos válidos" basados en un percentil de distancia para cada semilla.
            dist_thr_por_muestra = np.percentile(d_all, self.percentil_dist, axis=1).astype(float)
            # Cuenta cuántos de los K vecinos están dentro de su umbral de distancia.
            vecinos_validos_counts = np.sum(d_all <= dist_thr_por_muestra[:, None], axis=1).astype(int)

            # Meta agregada
            # Actualización de metadatos con resultados de filtrado.
            self._meta.update({
                "n_candidatas": int(len(X_min)),
                "n_filtradas": int(np.sum(comb)),
                "riesgo_medio": float(np.mean(riesgo[comb])) if np.any(comb) else None,
                "riesgo_std": float(np.std(riesgo[comb])) if np.any(comb) else None,
                "densidad_media": float(np.mean(densidades)) if densidades.size else None,
                "vecinos_validos_promedio": float(np.mean(vecinos_validos_counts)) if len(vecinos_validos_counts) else 0.0,
            })

            # Early-return si no hay suficientes filtradas para operar con k
            # Salida temprana: Número insuficiente de semillas válidas (filtradas) para calcular vecindades.
            if len(filtered_indices_local) < (K + 1):
                self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
                self._meta["k_efectivo"] = K
                # RETURN 3: Retorna X e y sin modificaciones si no hay suficientes semillas filtradas.
                # Razón: Las semillas válidas para la generación sintética son insuficientes para establecer 
                # un vecindario de tamaño K+1, lo que podría desestabilizar la generación sintética.
                return X.copy(), y.copy()

            # Cantidad a generar
            # Determinación del número de muestras sintéticas a generar.
            n_sint = int(max_sinteticas if max_sinteticas is not None else len(X_maj) - len(X_min))
            # Salida temprana: Si no hay muestras sintéticas a generar.
            if n_sint <= 0:
                self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
                self._meta["k_efectivo"] = K
                # RETURN 4: Retorna X e y sin modificaciones si n_sint <= 0.
                # Razón: No se necesita (o no se permite) generar muestras sintéticas.
                return X.copy(), y.copy()

            # Precálculos recortados
            # Se utilizan solo las semillas minoritarias que pasaron los filtros (pureza y densidad).
            X_min_filtrado = X_min[filtered_indices_local]
            vecinos_all_filtrado = vecinos_all_global[filtered_indices_local]
            dists_filtradas = d_all[filtered_indices_local]
            thr_filtradas = dist_thr_por_muestra[filtered_indices_local]

            # Inicialización de contadores para logging/diagnóstico
            gen_from_counts = defaultdict(int)
            last_delta_by_seed = {}
            last_neighbor_by_seed = {}

            rng = self.random_state
            lo, hi = self.DELTA_RANGE_INTERMEDIO

            muestras_sinteticas = []

            # Bucle de Generación de Muestras Sintéticas
            for _ in range(n_sint):
                # 1. Selección de la semilla (de X_min_filtrado)
                idx_loc = int(rng.randint(len(X_min_filtrado)))
                xi = X_min_filtrado[idx_loc]

                idxs_vec_all = vecinos_all_filtrado[idx_loc]
                dists = dists_filtradas[idx_loc]
                thr = thr_filtradas[idx_loc]

                # 2. Determinación de Vecinos Válidos
                # Se selecciona un vecino 'z' solo si su distancia 'd' a la semilla 'xi' está por debajo del umbral 'thr'.
                vecinos_validos = idxs_vec_all[dists <= thr]
                if len(vecinos_validos) == 0:
                    # Si no hay vecinos válidos (vecindario muy disperso), se salta la iteración.
                    continue

                # 3. Selección del vecino 'z' (de X)
                z_idx = int(rng.choice(vecinos_validos))
                xz = X[z_idx]

                # 4. Generación Sintética (Interpolación)
                # Se elige un delta aleatorio en el rango intermedio [lo, hi].
                delta = float(rng.uniform(lo, hi))
                # Nueva muestra sintética = xi + delta * (xz - xi) -> Se genera entre xi y xz.
                muestras_sinteticas.append(xi + delta * (xz - xi))

                # 5. Registro de Diagnóstico
                seed_global_idx = int(filtered_indices_global[idx_loc])
                gen_from_counts[seed_global_idx] += 1
                last_delta_by_seed[seed_global_idx] = delta
                last_neighbor_by_seed[seed_global_idx] = z_idx

            # Salida temprana: No se generó ninguna muestra sintética
            if not muestras_sinteticas:
                self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
                self._meta["k_efectivo"] = K
                # RETURN 5: Retorna X e y sin modificaciones si el bucle no pudo generar muestras.
                # Razón: Aunque el filtrado fue exitoso, ninguna de las semillas pudo encontrar un vecino "válido" 
                # (con distancia <= umbral) para realizar la interpolación.
                return X.copy(), y.copy()

            # Consolidación de resultados
            X_sint = np.asarray(muestras_sinteticas, dtype=float)
            y_sint = np.ones(len(X_sint), dtype=int)

            # Concatenación de muestras originales y sintéticas
            X_resampled = np.vstack([X, X_sint])
            y_resampled = np.hstack([y, y_sint])

            # Log por muestra (opcional; si no usás, podés omitir)
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

            # Finalización
            self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            self._meta["k_efectivo"] = K
            # RETURN 6 (FINAL): Retorna el dataset remuestreado exitosamente.
            return X_resampled, y_resampled

    def fit_resample_multiclass(self, X, y):
        """
        Oversampling por clase (one-vs-max) manteniendo todas las muestras originales.
        Respeta factor_equilibrio y topes. Logs robustos (k_efectivo seguro).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Identifico las clases
        clases = np.unique(y)
        # Inicializo resultados
        X_res = X.copy()
        y_res = y.copy()

        # Cuento el total de muestras originales
        total_original = len(y)
        # Cuento el total de muestras por clase
        conteo_original = Counter(y)
        # Identifico la clase mayoritaria
        max_count = max(conteo_original.values())

        self.meta_experimento = {
            "dataset": self.nombre_dataset,
            "k_neighbors": self.k,
            
            "percentil_dist": self.percentil_dist,
            "percentil_entropia": self.percentil_entropia,
            "percentil_densidad": self.percentil_densidad,
            "criterio_pureza": self.criterio_pureza,
            "factor_equilibrio": self.factor_equilibrio,
            "max_total_multiplier": self.max_total_multiplier,
            "max_sinteticas_por_clase": self.max_sinteticas_por_clase,
            "metric": self.metric,
            "random_state": self._loggable_random_state(),
            "timestamp": pd.Timestamp.now().isoformat()
        }

        # Por cada clase
        for clase in clases:
            # Aca se crea un problema binario: clase vs resto
            # Donde la clase actual coincida dentro de y, se va a colocar 1, y en el resto 0
            # Solo se va a sobremuestrear donde haya 1s
            y_bin = (y == clase).astype(int)
            # Cuento cuántas muestras hay de la clase actual
            actual = int(np.sum(y_bin))
            # De la clase con mas muestras, multiplico por el factor_equilibrio para 
            # saber cuántas muestras debería tener la clase actual
            objetivo = int(max_count * self.factor_equilibrio)
            # Si la cantidad actual es mayor o igual al objetivo, no se generan sintéticas
            estado = "sobremuestreada" if actual < objetivo else "no se sobremuestrea"
            # Calculo cuántas sintéticas harían falta para llegar al objetivo
            faltante_solicitado = max(0, objetivo - actual) # 400 - 200 = debo aumentar la clase actual a 200+
            faltante = max(0, objetivo - actual)

            tope_por_clase_aplicado = False
            tope_global_aplicado = False

            # Tope por clase
            # Se utiliza para limitar la cantidad de muestras sintéticas generadas por clase.
            # Esto se debe a que en algunos casos, una clase puede requerir muchas muestras sintéticas
            # para alcanzar el equilibrio deseado, lo que podría llevar a un sobreajuste o a un desequilibrio en el dataset.
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
                "n_filtradas": None,
                "k_efectivo": None,  # placeholder
            }

            if faltante > 0:
                sampler_tmp = PCSMOTE(
                    k_neighbors=self.k,
                    random_state=self._seed_init,
                    
                    percentil_dist=self.percentil_dist,
                    percentil_entropia=self.percentil_entropia,
                    percentil_densidad=self.percentil_densidad,

                    criterio_pureza=self.criterio_pureza,
                    factor_equilibrio=self.factor_equilibrio,
                    verbose=False,
                    max_total_multiplier=None,
                    max_sinteticas_por_clase=None,
                    guardar_distancias=self.guardar_distancias,
                    metric=self.metric
                )
                sampler_tmp.nombre_dataset = self.nombre_dataset

                X_bin_res, y_bin_res = sampler_tmp.fit_resample(X, y_bin, max_sinteticas=faltante)
                meta_clase_tmp = getattr(sampler_tmp, "_meta", {}) or {}

                # merge suave (mantiene claves por defecto si faltan en tmp)
                meta_clase.update({k: meta_clase_tmp.get(k, meta_clase[k]) for k in meta_clase.keys()})

                nuevos = len(X_bin_res) - len(X)
                if nuevos > 0:
                    X_nuevos = X_bin_res[-nuevos:]
                    y_nuevos = np.full(nuevos, clase)
                    X_res = np.vstack([X_res, X_nuevos])
                    y_res = np.hstack([y_res, y_nuevos])

                # Copiar logs por muestra agregando clase_objetivo (si los hay)
                for rec in getattr(sampler_tmp, "logs_por_muestra", []):
                    rec_copia = dict(rec)
                    rec_copia["clase_objetivo"] = clase
                    self.logs_por_muestra.append(rec_copia)

                # Motivo si no generó
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

            # k_efectivo robusto
            k_eff = meta_clase.get("k_efectivo", None)
            try:
                if k_eff is None or (isinstance(k_eff, float) and np.isnan(k_eff)):
                    k_eff = int(self.k)
                else:
                    k_eff = int(k_eff)
            except Exception:
                k_eff = int(self.k)

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
                "k_pedidos": int(self.k),
                "k_efectivo": int(k_eff),
                "percentil_densidad": self.percentil_densidad,
                "percentil_riesgo": self.percentil_dist,
                "criterio_pureza": self.criterio_pureza,
                "tecnica_sobremuestreo": "PCSMOTE",
                "factor_equilibrio": self.factor_equilibrio,
                "random_state": self._loggable_random_state(),

                "metric": self.metric,
                "timestamp": pd.Timestamp.now().isoformat(),
            })

        return X_res, y_res
