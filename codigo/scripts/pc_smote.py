import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state


class PCSMOTE:
    """
    PC-SMOTE (versión base, sin logs ni metadatos).

    Soporta:
    - Problemas binarios (0/1) mediante fit_resample_binario.
    - Problemas multiclase mediante esquema one-versus-all (OVA)
      en fit_resample_multiclass / fit_resample.

    Convención:
    - En la vista binaria OVA, la clase positiva es 1 (clase objetivo);
      el resto de clases se mapean a 0.

    PUREZA:
    --------
    Se controla con un único método:
        _calcular_pureza_por_muestra(y_binaria, indices_vecinos)

    que internamente aplica uno de dos criterios según self.criterio_pureza:

        - "proporcion":
            pureza_i = (# vecinos con y==1) / k_vecinos
            máscara: pureza_i >= umbral_pureza

        - "entropia":
            H_i = - Σ p_c log2 p_c  (c ∈ {0,1}), con H ∈ [0,1]
            máscara: H_i <= (1 - umbral_pureza)

            Es decir, si umbral_pureza = 0.8 (80% de misma clase en
            la visión por proporción), el criterio equivalente por
            entropía es H <= 0.2 (20% de incertidumbre).

    DENSIDAD:
    ---------
    - Primero se calcula un radio global u_densidad como el percentil
      percentil_dist_densidad de TODAS las distancias semilla→vecino
      (k vecinos globales de cada semilla positiva).
    - densidad_i = (# vecinos con dist <= u_densidad) / k_vecinos
    - Condición: densidad_i >= umbral_densidad.

    RIESGO:
    -------
    - Se calcula sobre los mismos k vecinos globales, pero usando un
      segundo radio global u_riesgo (percentil_dist_riesgo).
    - riesgo_i = (# vecinos de la clase contraria (0) con dist <= u_riesgo) / k_vecinos
    - Condición: riesgo_i <= umbral_riesgo.

    Una semilla positiva es CANDIDATA si:
        pureza_ok   AND densidad_ok AND riesgo_ok
    """

    DELTA_RANGO_INTERMEDIO = (0.4, 0.6)

    def __init__(
        self,
        k_vecinos=7,
        random_state=None,
        # percentiles (sobre distancias) para definir radios
        percentil_dist_densidad=80.0,
        percentil_dist_riesgo=40.0,
        # umbrales en proporción de k (para el criterio de proporción)
        umbral_pureza=0.60,
        umbral_densidad=0.80,
        umbral_riesgo=0.20,
        # criterio de pureza: "proporcion" o "entropia"
        criterio_pureza="proporcion",
        metric="euclidean",
        verbose=False,
    ):
        self.k_vecinos = int(k_vecinos)
        self.random_state = check_random_state(random_state)
        self.metric = str(metric)
        self.verbose = bool(verbose)

        self.percentil_dist_densidad = float(percentil_dist_densidad)
        self.percentil_dist_riesgo = float(percentil_dist_riesgo)

        self.umbral_pureza = float(umbral_pureza)
        self.umbral_densidad = float(umbral_densidad)
        self.umbral_riesgo = float(umbral_riesgo)

        criterio_pureza = str(criterio_pureza).lower()
        if criterio_pureza not in ("proporcion", "entropia"):
            raise ValueError(
                f"criterio_pureza debe ser 'proporcion' o 'entropia', "
                f"se recibió: {criterio_pureza}"
            )
        self.criterio_pureza = criterio_pureza

        self.X_sinteticas = None
        self.y_sinteticas = None

    # ------------------------------------------------------------------
    # PUREZA
    # ------------------------------------------------------------------

    def _calcular_pureza_por_proporcion(
        self, y_binaria, matriz_indices_vecinos
    ):
        """
        PUREZA por PROPORCIÓN:
            pureza_i = (# vecinos con y==1) / k_vecinos
        """
        y_binaria = np.asarray(y_binaria)
        cantidad_muestras = matriz_indices_vecinos.shape[0]
        purezas = np.zeros(cantidad_muestras, dtype=float)

        for indice_muestra in range(cantidad_muestras):
            indices_vecinos_actual = matriz_indices_vecinos[indice_muestra]
            cantidad_vecinos_misma_clase = 0

            for indice_vecino in indices_vecinos_actual:
                if int(y_binaria[indice_vecino]) == 1:
                    cantidad_vecinos_misma_clase += 1

            purezas[indice_muestra] = (
                cantidad_vecinos_misma_clase / float(self.k_vecinos)
            )

        return purezas

    def _calcular_pureza_por_entropia(
        self, y_binaria, matriz_indices_vecinos
    ):
        """
        PUREZA por ENTROPÍA (se devuelve H, NO 1-H):

            H_i = - Σ p_c log2(p_c), c ∈ {0,1}

        Donde:
            p_1 = (# vecinos con y==1) / k_vecinos
            p_0 = 1 - p_1

        Rango:
            - H_i = 0   → vecindario totalmente puro (sin incertidumbre)
            - H_i = 1   → vecindario 50/50 (máxima mezcla en binario)

        El filtro se aplica luego como:
            H_i <= (1 - umbral_pureza)
        """
        y_binaria = np.asarray(y_binaria)
        cantidad_muestras = matriz_indices_vecinos.shape[0]
        entropias = np.zeros(cantidad_muestras, dtype=float)

        for indice_muestra in range(cantidad_muestras):
            indices_vecinos_actual = matriz_indices_vecinos[indice_muestra]
            cantidad_vecinos_misma = 0

            for indice_vecino in indices_vecinos_actual:
                if int(y_binaria[indice_vecino]) == 1:
                    cantidad_vecinos_misma += 1

            p_misma = cantidad_vecinos_misma / float(self.k_vecinos)
            p_contraria = 1.0 - p_misma

            H = 0.0
            if p_misma > 0.0:
                H -= p_misma * np.log2(p_misma)
            if p_contraria > 0.0:
                H -= p_contraria * np.log2(p_contraria)

            entropias[indice_muestra] = H

        return entropias

    def _calcular_pureza_por_muestra(
        self, y_binaria, matriz_indices_vecinos
    ):
        """
        Punto único de entrada para calcular "pureza" según
        el criterio configurado.

        Si criterio_pureza == "proporcion":
            devuelve proporciones en [0,1] (mayor = más puro).

        Si criterio_pureza == "entropia":
            devuelve H en [0,1] (mayor = más mezclado).
            Luego la máscara de pureza se construye aparte.
        """
        if self.criterio_pureza == "proporcion":
            return self._calcular_pureza_por_proporcion(
                y_binaria, matriz_indices_vecinos
            )
        else:  # "entropia"
            return self._calcular_pureza_por_entropia(
                y_binaria, matriz_indices_vecinos
            )

    # ------------------------------------------------------------------
    # DENSIDAD y RIESGO
    # ------------------------------------------------------------------

    def _calcular_umbral_global_desde_distancias(
        self, matriz_distancias, percentil
    ):
        if matriz_distancias.size == 0:
            return 0.0
        distancias_vector = matriz_distancias.reshape(-1)
        umbral = float(np.percentile(distancias_vector, float(percentil)))
        return umbral

    def _calcular_densidad_por_muestra(
        self, matriz_distancias, umbral_densidad
    ):
        cantidad_muestras = matriz_distancias.shape[0]
        densidades = np.zeros(cantidad_muestras, dtype=float)

        for indice_muestra in range(cantidad_muestras):
            distancias_actual = matriz_distancias[indice_muestra]

            cantidad_vecinos_cercanos = 0
            for distancia_vecino in distancias_actual:
                if float(distancia_vecino) <= float(umbral_densidad):
                    cantidad_vecinos_cercanos += 1

            densidades[indice_muestra] = (
                cantidad_vecinos_cercanos / float(self.k_vecinos)
            )

        return densidades

    def _calcular_riesgo_por_muestra(
        self, y_binaria, matriz_indices_vecinos, matriz_distancias, umbral_riesgo
    ):
        y_binaria = np.asarray(y_binaria)
        cantidad_muestras = matriz_indices_vecinos.shape[0]
        riesgos = np.zeros(cantidad_muestras, dtype=float)

        for indice_muestra in range(cantidad_muestras):
            indices_vecinos_actual = matriz_indices_vecinos[indice_muestra]
            distancias_actual = matriz_distancias[indice_muestra]

            cantidad_vecinos_contrarios_cercanos = 0

            for posicion_vecino in range(len(indices_vecinos_actual)):
                indice_vecino = indices_vecinos_actual[posicion_vecino]
                distancia_vecino = distancias_actual[posicion_vecino]

                if float(distancia_vecino) <= float(umbral_riesgo):
                    if int(y_binaria[indice_vecino]) == 0:
                        cantidad_vecinos_contrarios_cercanos += 1

            riesgos[indice_muestra] = (
                cantidad_vecinos_contrarios_cercanos / float(self.k_vecinos)
            )

        return riesgos

    # ------------------------------------------------------------------
    # Núcleo binario (OVA internamente)
    # ------------------------------------------------------------------

    def _generar_sinteticas_binario(
        self,
        X,
        y_binaria,
        cantidad_sinteticas_objetivo,
    ):
        X = np.asarray(X, dtype=float)
        y_binaria = np.asarray(y_binaria)

        valores_unicos = np.unique(y_binaria)
        if not np.array_equal(np.sort(valores_unicos), np.array([0, 1])):
            raise ValueError(
                "En _generar_sinteticas_binario se espera y_binaria con valores {0,1}, "
                f"pero se encontró: {valores_unicos}"
            )

        indices_positivos = np.where(y_binaria == 1)[0]
        indices_negativos = np.where(y_binaria == 0)[0]

        cantidad_positivos = int(len(indices_positivos))

        if cantidad_sinteticas_objetivo <= 0:
            return None

        if cantidad_positivos < (self.k_vecinos + 1):
            return None

        # ----- vecindarios k-NN globales -----
        X_pos = X[indices_positivos]

        knn_global = NearestNeighbors(
            n_neighbors=self.k_vecinos + 1, metric=self.metric
        )
        knn_global.fit(X)

        distancias_todas, indices_vecinos_todos = knn_global.kneighbors(
            X_pos, return_distance=True
        )

        distancias_k = distancias_todas[:, 1:]        # (n_pos, k)
        indices_vecinos_k = indices_vecinos_todos[:, 1:]

        # ----- radios globales -----
        umbral_densidad = self._calcular_umbral_global_desde_distancias(
            distancias_k, self.percentil_dist_densidad
        )
        umbral_riesgo = self._calcular_umbral_global_desde_distancias(
            distancias_k, self.percentil_dist_riesgo
        )

        # ----- métricas por semilla -----
        valores_pureza = self._calcular_pureza_por_muestra(
            y_binaria, indices_vecinos_k
        )
        densidades = self._calcular_densidad_por_muestra(
            distancias_k, umbral_densidad
        )
        riesgos = self._calcular_riesgo_por_muestra(
            y_binaria, indices_vecinos_k, distancias_k, umbral_riesgo
        )

        # ----- máscaras -----
        if self.criterio_pureza == "proporcion":
            # valores_pureza = proporción de vecinos positivos
            mascara_pureza = valores_pureza >= self.umbral_pureza
        else:
            # valores_pureza = H (entropía). Para umbral_pureza = 0.8,
            # buscamos H <= 0.2
            umbral_H = 1.0 - self.umbral_pureza
            mascara_pureza = valores_pureza <= umbral_H

        mascara_densidad = densidades >= self.umbral_densidad
        mascara_riesgo = riesgos <= self.umbral_riesgo

        mascara_candidata = (
            mascara_pureza & mascara_densidad & mascara_riesgo
        )
        indices_locales_candidatas = np.where(mascara_candidata)[0]

        if len(indices_locales_candidatas) == 0:
            return None

        # ----- generación de sintéticas -----
        muestras_sinteticas = []
        rng = self.random_state
        delta_min, delta_max = self.DELTA_RANGO_INTERMEDIO

        for _ in range(cantidad_sinteticas_objetivo):
            indice_local_semilla = int(
                rng.choice(indices_locales_candidatas)
            )
            indice_global_semilla = int(
                indices_positivos[indice_local_semilla]
            )
            x_semilla = X[indice_global_semilla]

            indices_vecinos_actual = indices_vecinos_k[indice_local_semilla]
            distancias_actual = distancias_k[indice_local_semilla]

            # vecinos dentro de u_densidad
            vecinos_dentro_densidad = []
            for posicion_vecino in range(len(indices_vecinos_actual)):
                if float(distancias_actual[posicion_vecino]) <= float(
                    umbral_densidad
                ):
                    vecinos_dentro_densidad.append(
                        int(indices_vecinos_actual[posicion_vecino])
                    )

            # vecinos positivos dentro de ese radio
            vecinos_positivos_validos = []
            for indice_vecino in vecinos_dentro_densidad:
                if int(y_binaria[indice_vecino]) == 1:
                    vecinos_positivos_validos.append(indice_vecino)

            # fallback: cualquier vecino positivo del vecindario k
            if len(vecinos_positivos_validos) == 0:
                for indice_vecino in indices_vecinos_actual:
                    if int(y_binaria[indice_vecino]) == 1:
                        vecinos_positivos_validos.append(int(indice_vecino))

            if len(vecinos_positivos_validos) == 0:
                continue

            indice_vecino_elegido = int(rng.choice(vecinos_positivos_validos))
            x_vecino = X[indice_vecino_elegido]

            delta = float(rng.uniform(delta_min, delta_max))
            x_nueva = x_semilla + delta * (x_vecino - x_semilla)

            muestras_sinteticas.append(x_nueva)

        if len(muestras_sinteticas) == 0:
            return None

        X_sint = np.asarray(muestras_sinteticas, dtype=float)
        return X_sint

    # ------------------------------------------------------------------
    # Público binario
    # ------------------------------------------------------------------

    def fit_resample_binario(self, X, y_binaria, max_sinteticas=None):
        X = np.asarray(X, dtype=float)
        y_binaria = np.asarray(y_binaria)

        valores_unicos = np.unique(y_binaria)
        if not np.array_equal(np.sort(valores_unicos), np.array([0, 1])):
            raise ValueError(
                "fit_resample_binario espera y_binaria con valores {0,1}, "
                f"pero se encontró: {valores_unicos}"
            )

        indices_positivos = np.where(y_binaria == 1)[0]
        indices_negativos = np.where(y_binaria == 0)[0]

        cantidad_positivos = int(len(indices_positivos))
        cantidad_negativos = int(len(indices_negativos))

        if self.verbose:
            print(
                f"[PCSMOTE-binario] positivos={cantidad_positivos}, "
                f"negativos={cantidad_negativos}"
            )

        if max_sinteticas is None:
            deficit = cantidad_negativos - cantidad_positivos
            cantidad_sinteticas_objetivo = max(0, deficit)
        else:
            cantidad_sinteticas_objetivo = int(max_sinteticas)

        if cantidad_sinteticas_objetivo <= 0:
            self.X_sinteticas = None
            self.y_sinteticas = None
            return X.copy(), y_binaria.copy()

        X_sint = self._generar_sinteticas_binario(
            X, y_binaria, cantidad_sinteticas_objetivo
        )

        if X_sint is None or len(X_sint) == 0:
            self.X_sinteticas = None
            self.y_sinteticas = None
            return X.copy(), y_binaria.copy()

        y_sint = np.ones(len(X_sint), dtype=int)

        self.X_sinteticas = X_sint
        self.y_sinteticas = y_sint

        X_resampleado = np.vstack([X, X_sint])
        y_resampleado = np.hstack([y_binaria, y_sint])

        if self.verbose:
            print(
                f"[PCSMOTE-binario] sintéticas={len(X_sint)}, "
                f"nuevo_tamaño={len(y_resampleado)}"
            )

        return X_resampleado, y_resampleado

    # ------------------------------------------------------------------
    # Público multiclase OVA
    # ------------------------------------------------------------------

    def fit_resample_multiclass(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        clases_unicas, conteos = np.unique(y, return_counts=True)
        cantidad_clases = len(clases_unicas)

        if cantidad_clases < 2:
            raise ValueError(
                "fit_resample_multiclass requiere al menos 2 clases diferentes."
            )

        cantidad_maxima = int(np.max(conteos))

        if self.verbose:
            print(
                f"[PCSMOTE-multiclase] clases={clases_unicas}, "
                f"conteos={conteos}, max={cantidad_maxima}"
            )

        lista_X_sint = []
        lista_y_sint = []

        for indice_clase in range(cantidad_clases):
            etiqueta_clase = clases_unicas[indice_clase]
            conteo_clase = int(conteos[indice_clase])

            if conteo_clase >= cantidad_maxima:
                continue

            # y_binaria OVA
            y_binaria = np.zeros_like(y, dtype=int)
            for indice_muestra in range(len(y)):
                if y[indice_muestra] == etiqueta_clase:
                    y_binaria[indice_muestra] = 1
                else:
                    y_binaria[indice_muestra] = 0

            deficit_clase = cantidad_maxima - conteo_clase

            if self.verbose:
                print(
                    f"[PCSMOTE-multiclase] clase={etiqueta_clase}, "
                    f"conteo={conteo_clase}, deficit={deficit_clase}"
                )

            X_sint = self._generar_sinteticas_binario(
                X, y_binaria, deficit_clase
            )

            if X_sint is None or len(X_sint) == 0:
                continue

            y_sint = np.full(len(X_sint), etiqueta_clase)

            lista_X_sint.append(X_sint)
            lista_y_sint.append(y_sint)

        if len(lista_X_sint) == 0:
            self.X_sinteticas = None
            self.y_sinteticas = None
            return X.copy(), y.copy()

        X_sint_global = np.vstack(lista_X_sint)
        y_sint_global = np.hstack(lista_y_sint)

        self.X_sinteticas = X_sint_global
        self.y_sinteticas = y_sint_global

        X_resampleado = np.vstack([X, X_sint_global])
        y_resampleado = np.hstack([y, y_sint_global])

        if self.verbose:
            print(
                f"[PCSMOTE-multiclase] sintéticas_totales={len(X_sint_global)}, "
                f"nuevo_tamaño={len(y_resampleado)}"
            )

        return X_resampleado, y_resampleado

    def fit_resample(self, X, y):
        return self.fit_resample_multiclass(X, y)

    def obtener_sinteticas(self):
        return self.X_sinteticas, self.y_sinteticas
