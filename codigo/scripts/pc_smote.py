import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from Utils import Utils

class PCSMOTE(Utils):
    """
    PC-SMOTE (versión base, ahora con logging por muestra vía Utils).

    Soporta:
    - Problemas binarios (0/1) mediante fit_resample_binario.
    - Problemas multiclase mediante esquema one-versus-all (OVA)
      en fit_resample_multiclass / fit_resample.

    Convención:
    - En la vista binaria OVA, la clase positiva es 1 (clase objetivo);
      el resto de clases se mapean a 0.

    PUREZA:
    --------
    Se controla con 2 metodos 

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
        percentil_entropia=40.0,
        # umbrales en proporción de k (para el criterio de proporción)
        umbral_pureza=0.60,
        # umbral_densidad=0.80,
        # umbral_riesgo=0.20,
        # criterio de pureza: "proporcion" o "entropia"
        criterio_pureza="proporcion",
        metric="euclidean",
        verbose=False,
    ):
        super().__init__()

        self.k_vecinos = int(k_vecinos)
        self.random_state = check_random_state(random_state)
        self.metric = str(metric)
        self.verbose = bool(verbose)

        self.percentil_dist_densidad = float(percentil_dist_densidad)
        self.percentil_dist_riesgo = float(percentil_dist_riesgo)
        self.percentil_entropia = float(percentil_entropia)

        self.umbral_pureza = float(umbral_pureza)
        self.umbral_densidad = None
        self.umbral_riesgo = None

        self.entropias = None
        self.densidades = None
        self.riesgos = None
        self.mascara_vecino_minoritario = None
        self.mascara_entropia_baja = None
        self.mascara_pureza = None  

        # --- acumuladores globales (todas las clases OVA) ---
        self.mascara_entropia_baja_global = None
        self.mascara_vecino_minoritario_global = None
        self.mascara_pureza_global = None
        self.densidades_global = None
        self.riesgos_global = None
        self.metricas_por_clase = []

        criterio_pureza = str(criterio_pureza).lower()
        if criterio_pureza not in ("proporcion", "entropia"):
            raise ValueError(
                f"criterio_pureza debe ser 'proporcion' o 'entropia', "
                f"se recibió: {criterio_pureza}"
            )
        self.criterio_pureza = criterio_pureza

        # nombre de configuración (para el log)
        self.nombre_configuracion = (
            f"D{int(self.percentil_dist_densidad)}_"
            f"R{int(self.percentil_dist_riesgo)}_"
            f"P{self.criterio_pureza}"
        )

        self.X_sinteticas = None
        self.y_sinteticas = None
        
    def _reiniciar_metricas_por_clase(self):
        self.metricas_por_clase = []

    def _acumular_metricas_por_clase(
        self,
        clase_objetivo,
        mascara_entropia_baja,
        mascara_vecino_minoritario,
        mascara_pureza,
        densidades,
        riesgos,
    ):
        registro = {
            "clase_objetivo": clase_objetivo,
            "mascara_entropia_baja": None if mascara_entropia_baja is None else mascara_entropia_baja.astype(float),
            "mascara_vecino_minoritario": None if mascara_vecino_minoritario is None else mascara_vecino_minoritario.astype(float),
            "mascara_pureza": mascara_pureza.astype(float) if mascara_pureza is not None else None,
            "densidades": densidades.astype(float),
            "riesgos": riesgos.astype(float),
        }
        self.metricas_por_clase.append(registro)

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

    # ------------------------------------------------------------------
    # DENSIDAD y RIESGO
    # ------------------------------------------------------------------

    def _calcular_umbral_global_desde_distancias(self, matriz_distancias, percentil):

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
    # Núcleo binario (OVA internamente) + LOG
    # ------------------------------------------------------------------

    def _generar_sinteticas_binario(
        self,
        X,
        y_binaria,
        cantidad_sinteticas_objetivo,
        y_original=None,
        clase_objetivo=1,
    ):
        """
        Genera sintéticas para problema binario y, además, loguea por semilla.

        - y_binaria ∈ {0,1}.
        - y_original: etiquetas originales (multiclase). Si es None, usa y_binaria.
        - clase_objetivo: etiqueta real usada en OVA (para el log).
        """
        X = np.asarray(X, dtype=float)
        y_binaria = np.asarray(y_binaria)

        if y_original is None:
            y_original = y_binaria

        valores_unicos = np.unique(y_binaria)
        if not np.array_equal(np.sort(valores_unicos), np.array([0, 1])):
            raise ValueError(
                "En _generar_sinteticas_binario se espera y_binaria con valores {0,1}, "
                f"pero se encontró: {valores_unicos}"
            )

        indices_positivos = np.where(y_binaria == 1)[0]

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
        indices_vecinos_k = indices_vecinos_todos[:, 1:]  # (n_pos, k)

        # ----- radios globales -----
        umbral_densidad = self._calcular_umbral_global_desde_distancias(
            distancias_k, self.percentil_dist_densidad
        )
        umbral_riesgo = self._calcular_umbral_global_desde_distancias(
            distancias_k, self.percentil_dist_riesgo
        )

        # ----- métricas por semilla -----
        # Queremos tener SIEMPRE, al menos:
        # - proporciones_min: proporción de vecinos minoritarios (p_misma)
        # - entropias: H (solo si criterio_pureza = "entropia")
        proporciones_min = None
        entropias = None

        proporciones_min = self._calcular_pureza_por_proporcion(
            y_binaria, indices_vecinos_k
        )

        if self.criterio_pureza == "proporcion":
            # En este caso, pureza = proporción de vecinos minoritarios
            self.proporciones_min = proporciones_min  # en [0,1]
        else:
            # criterio_pureza == "entropia"
            # 1) Entropía del vecindario (mezcla)
            entropias = self._calcular_pureza_por_entropia(
                y_binaria, matriz_indices_vecinos=indices_vecinos_k
            )
            

        densidades = self._calcular_densidad_por_muestra(
            distancias_k, umbral_densidad
        )

        self.densidades = densidades

        riesgos = self._calcular_riesgo_por_muestra(
            y_binaria, indices_vecinos_k, distancias_k, umbral_riesgo
        )

        self.riesgos = riesgos

        UMBRAL_DOMINANCIA_MINORITARIA = 0.5
        # ----- máscaras -----
        if self.criterio_pureza == "proporcion":
            # Igual que antes: pureza = proporción de minoritarios >= umbral
            mascara_pureza = proporciones_min >= self.umbral_pureza
            umbral_entropia = None
            mascara_entropia_baja = None
            mascara_vecino_minoritario = None
        else:
            # criterio_pureza = "entropia"
            umbral_entropia = float(
                np.percentile(entropias, float(self.percentil_entropia))
            )

            # condición 1: baja mezcla (entropía baja)
            mascara_entropia_baja = entropias <= umbral_entropia
            self.mascara_entropia_baja = mascara_entropia_baja
            # condición 2: las minoritarias siempre le tienen que ganar a las mayoritarias
            # asi me aseguro que cuando hago el AND con mascara_entropia_baja, me quedo con
            # muestras de baja mezcla donde domina la minoritaria
            mascara_vecino_minoritario = proporciones_min > UMBRAL_DOMINANCIA_MINORITARIA
            self.mascara_vecino_minoritario = mascara_vecino_minoritario

            # pureza final: vecindario poco mezclado Y con presencia de minoritarios
            mascara_pureza = mascara_entropia_baja & mascara_vecino_minoritario
            self.mascara_pureza = mascara_pureza

        mascara_densidad = densidades >= umbral_densidad
        mascara_riesgo = riesgos <= umbral_riesgo

        # Al final de _generar_sinteticas_binario, justo antes del return X_sint

        if not hasattr(self, "metricas_por_clase"):
            self.metricas_por_clase = []

        registro_clase = {
            "clase": clase_objetivo,
            "mascara_entropia_baja": self.mascara_entropia_baja.astype(float)
                                    if self.mascara_entropia_baja is not None else None,
            "mascara_vecino_minoritario": self.mascara_vecino_minoritario.astype(float)
                                        if self.mascara_vecino_minoritario is not None else None,
            "mascara_pureza": self.mascara_pureza.astype(float)
                            if self.mascara_pureza is not None else None,
            "densidades": densidades.astype(float),
            "riesgos": riesgos.astype(float),
            "entropias": entropias.astype(float) if entropias is not None else None,
        }

        self.metricas_por_clase.append(registro_clase)


        mascara_candidata = (
            mascara_pureza & mascara_densidad & mascara_riesgo
        )

        indices_locales_candidatas = np.where(mascara_candidata)[0]

        # ----- generación de sintéticas -----
        muestras_sinteticas = []
        rng = self.random_state
        delta_min, delta_max = self.DELTA_RANGO_INTERMEDIO

        cantidad_semillas_pos = len(indices_positivos)
        conteo_sinteticas_por_semilla = np.zeros(
            cantidad_semillas_pos, dtype=int
        )

        for _ in range(cantidad_sinteticas_objetivo):
            if len(indices_locales_candidatas) == 0:
                break

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
            conteo_sinteticas_por_semilla[indice_local_semilla] += 1

        if len(muestras_sinteticas) == 0:
            X_sint = None
        else:
            X_sint = np.asarray(muestras_sinteticas, dtype=float)

        # ------------------------------------------------------------------
        # LOG POR MUESTRA (migrado a Utils)
        # ------------------------------------------------------------------
        self.loguear_semillas_positivas(
            nombre_configuracion=self.nombre_configuracion,
            clase_objetivo=clase_objetivo,
            y_original=y_original,
            y_binaria=y_binaria,
            k=self.k_vecinos,
            indices_positivos=indices_positivos,
            indices_vecinos_k=indices_vecinos_k,
            distancias_k=distancias_k,
            umbral_densidad=umbral_densidad,
            umbral_riesgo=umbral_riesgo,
            umbral_entropia=umbral_entropia,
            criterio_pureza=self.criterio_pureza,
            proporciones_min=proporciones_min,
            densidades=densidades,
            riesgos=riesgos,
            entropias=entropias,
            mascara_pureza=mascara_pureza,
            mascara_densidad=mascara_densidad,
            mascara_riesgo=mascara_riesgo,
            mascara_candidata=mascara_candidata,
            conteo_sinteticas_por_semilla=conteo_sinteticas_por_semilla,
        )

        return X_sint

    # ------------------------------------------------------------------
    # Público binario
    # ------------------------------------------------------------------

    def fit_resample_binario(self, X, y_binaria, max_sinteticas=None):
        X = np.asarray(X, dtype=float)
        y_binaria = np.asarray(y_binaria)

        # reset del log para esta llamada
        self.logs_por_muestra = []

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
            X, y_binaria, cantidad_sinteticas_objetivo,
            y_original=y_binaria,
            clase_objetivo=1,
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

        # reset del log para esta llamada global (todas las clases)
        self.logs_por_muestra = []

        self._reiniciar_metricas_por_clase()

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
                X, y_binaria, deficit_clase,
                y_original=y,
                clase_objetivo=etiqueta_clase,
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
