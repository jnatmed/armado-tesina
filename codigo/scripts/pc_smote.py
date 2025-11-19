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
from meta_pcsmote import GeneradorMetaPCSMOTE

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
    X_syn, y_syn = None, None # X_syn, y_syn

    # Acumulador GLOBAL de logs por muestra, agrupado por nombre de dataset.
    # Clave: nombre_dataset  ->  Valor: DataFrame con todas las filas acumuladas
    _acumulador_logs_por_muestra = {}

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
                 percentil_riesgo,
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

        # Percentil de riesgo 
        if percentil_riesgo is None:
            raise ValueError("percentil_riesgo es obligatorio y no puede ser None.")

        self.percentil_riesgo = float(percentil_riesgo)
        print('self.riesgo recibido en el construct: ', self.percentil_riesgo)


        self._X_syn = None
        self._y_syn = None

        # --- META aislada por composición ---
        self._X_res_meta = None
        self._y_res_meta = None

        # helper de meta (aislado)
        self._generador_meta = GeneradorMetaPCSMOTE()

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

    def snapshot(self):
        return {
            "k": self.k,
            "random_state": self._seed_init,
            "percentil_dist": self.percentil_dist,
            "percentil_densidad": self.percentil_densidad,
            "percentil_entropia": self.percentil_entropia,
            "percentil_riesgo": self.percentil_riesgo,
            "criterio_pureza": self.criterio_pureza,
            "factor_equilibrio": self.factor_equilibrio,
            "metric": self.metric,
            "guardar_distancias": self.guardar_distancias,
            "max_total_multiplier": self.max_total_multiplier,
            "max_sinteticas_por_clase": self.max_sinteticas_por_clase,
            "nombre_dataset": self.nombre_dataset,
            "nombre_configuracion": getattr(self, "nombre_configuracion", None)
        }
        

    # ─────────────────── Acumulador de logs POR MUESTRA ───────────────────
    # Clave: nombre_dataset → lista de DataFrames de logs (uno por configuración).
    _acumulador_logs_por_muestra_por_dataset = {}

    @classmethod
    def acumular_logs_por_muestra_por_dataset(cls, nombre_dataset, df_logs):
        """
        Acumula los logs por muestra de una configuración en memoria,
        agrupados por nombre de dataset.
        """
        if not nombre_dataset:
            # Si no hay dataset definido, no acumulamos de forma especial.
            return

        if nombre_dataset not in cls._acumulador_logs_por_muestra_por_dataset:
            cls._acumulador_logs_por_muestra_por_dataset[nombre_dataset] = []

        cls._acumulador_logs_por_muestra_por_dataset[nombre_dataset].append(df_logs)

    @classmethod
    def obtener_logs_por_muestra_acumulados(cls, nombre_dataset):
        """
        Devuelve un DataFrame con todos los logs acumulados para un dataset.
        Si no hay nada acumulado, devuelve el DataFrame vacío.
        """
        import pandas as pd  # import local para no romper dependencias

        lista = cls._acumulador_logs_por_muestra_por_dataset.get(nombre_dataset, [])
        if not lista:
            return pd.DataFrame()

        return pd.concat(lista, ignore_index=True)

    # ------------ setters y getters ------------- # 

    # setters/getters
    def setX_syn(self, X_syn): self._X_syn = None if X_syn is None else np.asarray(X_syn)
    def setY_syn(self, y_syn): self._y_syn = None if y_syn is None else np.asarray(y_syn)
    def getX_syn(self): return self._X_syn
    def getY_syn(self): return self._y_syn
    def get_sinteticas(self): return self._X_syn, self._y_syn
    # --------- GETTERS de salida meta (no alteran returns) ---------
    def get_X_res_meta(self):
        """Devuelve la última X resampleada con columnas meta anexadas (o None)."""
        return self._X_res_meta

    def get_y_res_meta(self):
        """Devuelve la última y resampleada asociada a X_res_meta (o None)."""
        return self._y_res_meta    

    # ------------------------------- Densidad por intersección -------------------------------

    def calcular_densidad_interseccion(self, X_min, vecinos_local, dists_min_local):
        """
        Densidad por intersección entre semillas MINORITARIAS (versión corregida):

          - Primero se arma un vector global con TODAS las distancias entre semillas
            minoritarias y sus vecinos minoritarios.
          - Se define un ÚNICO umbral global u_global = percentil(self.percentil_dist)
            de ese conjunto global de distancias.
          - Para cada semilla i, la densidad es la fracción de sus vecinos minoritarios
            cuya distancia a la semilla es <= u_global.

        Esto evita que cada semilla tenga su propio percentil local (que colapsaba
        las densidades alrededor de un valor fijo) y permite que la densidad
        realmente discrimine semillas más “apretadas” de las más aisladas.
        """
        X_min = np.asarray(X_min)
        n_min = len(X_min)
        if n_min == 0:
            return np.array([], dtype=float)

        densidades = np.zeros(n_min, dtype=float)
        self._diag_densidad = {"semillas_con_hits": 0, "total_hits": 0}

        # 1) Construir vector global de distancias minoritarias
        distancias_globales = []
        for i in range(n_min):
            # dists_min_local[i] ya debería contener las distancias desde la semilla i
            # a sus vecinos minoritarios (en el mismo orden que vecinos_local[i])
            if i >= len(dists_min_local):
                continue

            d_i = dists_min_local[i]
            if d_i is None:
                continue

            d_i = np.asarray(d_i, dtype=float)
            if d_i.size == 0:
                continue

            # Ignoro ceros (la propia semilla) para no contaminar el percentil
            for valor in d_i:
                if valor > 0.0:
                    distancias_globales.append(float(valor))

        if len(distancias_globales) == 0:
            # No hay distancias válidas -> densidades todas en 0
            return densidades

        distancias_globales = np.asarray(distancias_globales, dtype=float)

        # 2) Umbral global según percentil_dist (o mediana si no se configuró)
        if self.percentil_dist is not None:
            umbral_global = float(np.percentile(distancias_globales, self.percentil_dist))
        else:
            umbral_global = float(np.percentile(distancias_globales, 50.0))

        # Guardo el umbral global de densidad para diagnósticos
        self._meta["umbral_densidad_global"] = umbral_global

        # 3) Calcular densidad por semilla usando el umbral global
        for i in range(n_min):
            if i >= len(vecinos_local):
                continue

            indices_vecinos_locales = vecinos_local[i]
            if len(indices_vecinos_locales) == 0:
                densidades[i] = 0.0
                continue

            if i >= len(dists_min_local):
                densidades[i] = 0.0
                continue

            d_i = np.asarray(dists_min_local[i], dtype=float)
            if d_i.size == 0:
                densidades[i] = 0.0
                continue

            # Vecinos que están a distancia menor o igual al umbral global
            intersecciones = int(np.sum(d_i <= umbral_global))

            if intersecciones > 0:
                # semillas_con_hits: cuántas semillas tienen al menos un vecino “cercano”
                self._diag_densidad["semillas_con_hits"] += 1
                # total_hits: suma de todos los vecinos “cercanos” en el dataset
                self._diag_densidad["total_hits"] += intersecciones

            # Densidad = fracción de vecinos locales que caen dentro del umbral global
            densidades[i] = intersecciones / float(len(indices_vecinos_locales))

        return densidades


    def calcular_entropia(self, vecinos_all_global, y):
        """
        Entropía NORMALIZADA para subproblemas OVA (One-vs-All), base 2.

        En OVA los vecindarios SIEMPRE son binarios (clase positiva vs. negativa).
        Por lo tanto:
        H_max = log2(2) = 1.0

        Interpretación:
        - entropía = 0.0  -> vecindario completamente puro (solo una clase)
        - entropía = 1.0  -> vecindario totalmente mezclado (50/50 entre 0 y 1)
        """

        entropias = []

        y = np.asarray(y)
        H_max = 1.0   # porque OVA siempre es binario

        for idxs in vecinos_all_global:
            etiquetas_vecindario = y[idxs]

            clases, counts = np.unique(etiquetas_vecindario, return_counts=True)
            p = counts / counts.sum()

            H = float(entropy(p, base=2))   # entropía real binaria

            # normalización trivial en OVA: dividir por 1.0
            H_normalizada = H / H_max

            entropias.append(H_normalizada)

        return np.array(entropias, dtype=float)

    """
    calcula el riesgo de cada vecino (proporcion de vecinos mayoritarios)
    para determinar aquellos que se encuentran en la frontera o no
    """
    def calcular_riesgo(self, vecinos_all_global, y, K):
        riesgo = []
        for lista_vecinos in vecinos_all_global:   # vecinos_all_global es una lista con los índices de los K vecinos de cada muestra
            etiquetas_vecinos = y[lista_vecinos]   # obtenemos las etiquetas de esos K vecinos
            cantidad_mayoritarios = np.sum(etiquetas_vecinos == 0)  # contamos cuántos son clase 0 (mayoritaria)
            proporcion_mayoritarios = cantidad_mayoritarios / K     # calculamos la proporción
            riesgo.append(proporcion_mayoritarios)                  # guardamos el valor para esa muestra

        # devuelvo una lista con el riesgo de cada muestra
        # xejemplo : [0.857, 0.714, 0.429, ...]
        # siendo riesgo del mismo tamaño que vecinos_all_global
        # cada posicion corresponde al riesgo de la muestra en esa posicion
        return np.array(riesgo, dtype=float)        

    def aplicar_filtro_por_riesgo(self, vector_riesgo):
        """
        Aplica el filtro por riesgo usando un percentil (np.percentile) pero con
        pasos explícitos y bucles para la máscara.

        Entradas:
        - vector_riesgo: array-like de floats en [0,1]
        - self.percentil_riesgo: percentil en [0,100] o None

        Salida:
        - mascara_riesgo: np.ndarray[bool] del mismo tamaño que vector_riesgo

        Complejidad:
        - Cálculo de percentil (NumPy): O(n log n) por el ordenamiento interno
        - Construcción de máscara explícita: O(n)
        """
        import numpy as np

        # Normalización de entrada
        vector_riesgo = np.asarray(vector_riesgo, dtype=float)

        # Caso: vector vacío
        if vector_riesgo.size == 0:
            self._meta["umbral_riesgo_min"] = None
            print("⚠️ No hay muestras para configurar riesgo.")
            return np.zeros(0, dtype=bool)

        # Caso: sin percentil configurado -> aceptar todas las muestras
        if self.percentil_riesgo is None:
            mascara_riesgo = np.zeros(vector_riesgo.shape[0], dtype=bool)
            # Bucle explícito para visibilidad
            indice = 0
            while indice < vector_riesgo.shape[0]:
                mascara_riesgo[indice] = True
                indice += 1
            self._meta["umbral_riesgo_min"] = None
            print("⚠️ No hay percentil riesgo configurado.")
            return mascara_riesgo

        # 1) Umbral por percentil (mantenemos np.percentile, como pediste)
        percentil_configurado = float(self.percentil_riesgo)
        if percentil_configurado < 0.0:
            percentil_configurado = 0.0
        if percentil_configurado > 100.0:
            percentil_configurado = 100.0

        umbral_riesgo_minimo = float(np.percentile(vector_riesgo, percentil_configurado))
        print("Umbral por percentil riesgo:", umbral_riesgo_minimo)
        self._meta["umbral_riesgo_min"] = umbral_riesgo_minimo

        # 2) Construcción explícita de la máscara
        """
        crea un array de NumPy del mismo tamaño que vector_riesgo, lleno de ceros lógicos (False), 
        que después se va a ir completando con True o False según el riesgo de cada muestra
        """
        mascara_riesgo = np.zeros(vector_riesgo.shape[0], dtype=bool)

        for indice, valor_riesgo_actual in enumerate(vector_riesgo):
            if valor_riesgo_actual >= umbral_riesgo_minimo:
                mascara_riesgo[indice] = True
            else:
                mascara_riesgo[indice] = False

        return mascara_riesgo

    def calcular_proporciones_minoritarios_en_vecindario(self, y, matriz_vecinos_indices, cantidad_vecinos_K):
        """
        Calcula la proporción de vecinos minoritarios (etiqueta == 1) para cada muestra.
        """
        import numpy as np

        n_muestras = matriz_vecinos_indices.shape[0]
        proporciones_minoritarios = np.zeros(n_muestras, dtype=float)

        for indice_muestra, vecinos_de_muestra in enumerate(matriz_vecinos_indices):
            cantidad_minoritarios = 0 # contador de vecinos minoritarios
            for indice_vecino in vecinos_de_muestra: # por cada vecino de la muestra
                # Contar vecinos minoritarios
                if int(y[indice_vecino]) == 1:
                    cantidad_minoritarios += 1
            """
            Calcula la proporción de vecinos minoritarios dividiendo 
            la cantidad de vecinos minoritarios por la cantidad total de vecinos K.
            """
            proporciones_minoritarios[indice_muestra] = cantidad_minoritarios / float(cantidad_vecinos_K)

        return proporciones_minoritarios

    def construir_mascara_pureza_por_proporcion(self, proporciones_minoritarios, cantidad_vecinos_K):
        """
        Crea una máscara booleana para marcar muestras con mezcla de clases
        (es decir, vecindarios que tienen al menos 1 minoritario y 1 mayoritario).
        """
        import numpy as np

        n_muestras = len(proporciones_minoritarios)
        mascara_pureza_proporcion = np.zeros(n_muestras, dtype=bool)

        epsilon = 1.0 / float(cantidad_vecinos_K)
        # limite inferior
        limite_inferior = epsilon
        # limite superior
        limite_superior = 1.0 - epsilon

        if hasattr(self, "_meta"):
            self._meta["pureza_eps"] = epsilon
            self._meta["pureza_limite_inferior"] = limite_inferior
            self._meta["pureza_limite_superior"] = limite_superior

        for indice, proporcion_actual in enumerate(proporciones_minoritarios):
            """
            solo marco como True aquellas muestras cuya proporción de vecinos minoritarios
            esté entre el límite inferior y el límite superior, es decir, aquellas muestras
            que tienen mezcla de clases en su vecindario
            """
            mascara_pureza_proporcion[indice] = (
                limite_inferior <= proporcion_actual <= limite_superior
            )

        return mascara_pureza_proporcion

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
            # preservar formato de entrada para X_res_meta (DataFrame vs ndarray)
            self._generador_meta.detectar_formato_entrada(X)

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
            riesgo = self.calcular_riesgo(vecinos_all_global, y, K)

            # --- Filtro por RIESGO (por percentil) ---
            mask_riesgo = self.aplicar_filtro_por_riesgo(riesgo)

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
                # Entropía normalizada del vecindario:
                #   0.0 => vecindario puro
                #   1.0 => vecindario fuertemente mezclado (frontera)
                entropias = self.calcular_entropia(vecinos_all_global, y)

                if self.percentil_entropia is not None:
                    # umb_ent: umbral de entropía basado en el percentil configurado.
                    umb_ent = float(np.percentile(entropias, self.percentil_entropia))
                    # Para frontera me quedo con las muestras con entropía MAYOR o IGUAL al umbral.
                    pureza_mask = entropias >= umb_ent
                else:
                    umb_ent = None
                    # Si no se define percentil, considero como candidatas las muestras con algo de mezcla
                    # (entropía > 0) y descarto las completamente puras.
                    pureza_mask = entropias > 0.0

                self._meta["umbral_entropia"] = umb_ent

            elif self.criterio_pureza == 'proporcion':
                proporciones_min = self.calcular_proporciones_minoritarios_en_vecindario(
                    y, vecinos_all_global, K
                )
                pureza_mask = self.construir_mascara_pureza_por_proporcion(
                    proporciones_min, K
                )

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

            # Combinación de filtros (pureza Y densidad Y riesgo)
            # Estrategia: solo considero semillas que están
            #   - en zona de frontera (pureza/frontera verdadera),
            #   - en zona densa (densidad suficiente),
            #   - y con riesgo alto (proporción de mayoritarios alta).
            #
            # Es decir, TODAS las condiciones deben cumplirse.
            comb = pureza_mask & densidad_mask & mask_riesgo

            filtered_indices_local = np.where(comb)[0]  # Índices locales sobre X_min
            filtered_indices_global = idxs_min_global[filtered_indices_local]  # Índices globales sobre X

            # ------------------- Diagnóstico de vecinos válidos (umbral coherente) -------------------
            # Intento usar el mismo umbral_global que se usó en calcular_densidad_interseccion.
            umbral_global = self._meta.get("umbral_densidad_global", None)

            if umbral_global is not None:
                # Caso ideal: uso un ÚNICO umbral global para todo el dataset.
                dist_thr_por_muestra = np.full(d_all.shape[0], float(umbral_global), dtype=float)
                vecinos_validos_counts = np.sum(d_all <= umbral_global, axis=1).astype(int)
            else:
                # Fallback: si por algún motivo no se configuró umbral_densidad_global,
                # reproduzco el comportamiento anterior (percentil por muestra).
                dist_thr_por_muestra = np.percentile(d_all, self.percentil_dist, axis=1).astype(float)
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
                #    Se usa el mismo umbral (global o fallback) que se utilizó en el diagnóstico
                #    de vecinos válidos y que es coherente con el cálculo de densidad.
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

            self.setX_syn(X_sint)
            self.setY_syn(y_sint)

            # Concatenación de muestras originales y sintéticas
            X_resampled = np.vstack([X, X_sint])
            y_resampled = np.hstack([y, y_sint])

            print("self_meta:", self._meta)

            # Log por muestra (opcional; si no usás, podés omitir)
            for i in range(len(X_min)):
                self._log_muestra(
                    i, X, X_min, y, idxs_min_global,
                    comb, riesgo, densidades, entropias, proporciones_min,
                    pureza_mask, densidad_mask, mask_riesgo,
                    umb_ent, umb_den,
                    vecinos_all_global, vecinos_min_global,
                    vecinos_validos_counts, dist_thr_por_muestra,
                    gen_from_counts, last_delta_by_seed, last_neighbor_by_seed, self._meta
                )

            # Finalización
            self._meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            self._meta["k_efectivo"] = K
            # RETURN 6 (FINAL): Retorna el dataset remuestreado exitosamente.
            # construir y guardar X_res_meta/y_res_meta sin alterar el return
            self._X_res_meta, self._y_res_meta = self._generador_meta.construir_X_y_con_meta(
                X_res=X_resampled,
                y_res=y_resampled,
                K=self.k,
                metric=self.metric,
                criterio_pureza=self.criterio_pureza,
                fn_densidad_interseccion=self.calcular_densidad_interseccion
            )
            return X_resampled, y_resampled

    def fit_resample_multiclass(self, X, y):
        """
        Oversampling por clase (one-vs-max) manteniendo todas las muestras originales.
        Respeta factor_equilibrio y topes. Logs robustos (k_efectivo seguro).
        """
        self._generador_meta.detectar_formato_entrada(X)

        X = np.asarray(X)
        y = np.asarray(y)

        # --- acumuladores globales de sintéticas ---
        X_syn_all = []
        y_syn_all = []
        
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
                    percentil_riesgo=self.percentil_riesgo,  

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

                    # --- acumular sintéticas para graficar ---
                    X_tmp_syn = sampler_tmp.getX_syn()
                    y_tmp_syn = sampler_tmp.getY_syn()
                    if X_tmp_syn is not None and len(X_tmp_syn) > 0:
                        X_syn_all.append(X_tmp_syn)
                        # Etiquetar con la CLASE real (no binaria)
                        y_syn_all.append(np.full(len(X_tmp_syn), clase))


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
                "percentil_riesgo": self.percentil_riesgo,
                "criterio_pureza": self.criterio_pureza,
                "tecnica_sobremuestreo": "PCSMOTE",
                "factor_equilibrio": self.factor_equilibrio,
                "random_state": self._loggable_random_state(),

                "metric": self.metric,
                "timestamp": pd.Timestamp.now().isoformat(),
            })

            # --- NUEVO: setear en el sampler padre lo acumulado ---
        if len(X_syn_all) > 0:
            self.setX_syn(np.vstack(X_syn_all))
            self.setY_syn(np.hstack(y_syn_all))
        else:
            self.setX_syn(None)
            self.setY_syn(None)

        self._X_res_meta, self._y_res_meta = self._generador_meta.construir_X_y_con_meta(
            X_res=X_res,
            y_res=y_res,
            K=self.k,
            metric=self.metric,
            criterio_pureza=self.criterio_pureza,
            fn_densidad_interseccion=self.calcular_densidad_interseccion
        )
        return X_res, y_res
