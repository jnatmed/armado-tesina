from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
import numpy as np

class PCSMOTE:
    def __init__(self, sampling_strategy='auto', k_neighbors=5, random_state=None, radio_densidad=1.0):
        """
        Constructor del algoritmo PC-SMOTE con análisis de densidad local.
        
        Parámetros:
        - sampling_strategy: estrategia de muestreo (por ahora no utilizada)
        - k_neighbors: número de vecinos para evaluar riesgo y densidad
        - random_state: para reproducibilidad
        - radio_densidad: radio para definir las áreas circulares de intersección
        """
        self.sampling_strategy = sampling_strategy
        self.k = k_neighbors
        self.random_state = check_random_state(random_state)
        self.radio_densidad = radio_densidad

    def calcular_densidad_interseccion(self, X_min, vecinos, radio):
        """
        Calcula la densidad local como la cantidad de vecinos que se intersecan
        espacialmente con la muestra, según un radio circular fijo.

        Densidad = intersecciones / k

        Si ninguna área vecina se superpone, se considera "ruido" (densidad = 0).
        """
        densidades = []
        for i, xi in enumerate(X_min):
            intersecciones = 0
            for j in vecinos[i]:
                xj = X_min[j]
                distancia = np.linalg.norm(xi - xj)
                if distancia <= 2 * radio:
                    intersecciones += 1
            densidad = intersecciones / len(vecinos[i])
            densidades.append(densidad)
        return np.array(densidades)

    def fit_resample(self, X, y):
        """
        Aplica PC-SMOTE con filtrado por riesgo y densidad local.

        Paso 1: calcula el riesgo basado en proporción de vecinos mayoritarios.
        Paso 2: elimina muestras minoritarias con densidad nula (ruido).
        Paso 3: genera muestras sintéticas solo en regiones densas o semi-densas.
        """
        X = np.array(X)
        y = np.array(y)

        # Separar las clases
        X_min = X[y == 1]  # Minoritaria
        X_maj = X[y == 0]  # Mayoritaria

        # === Fase 1: Riesgo ===
        # Evaluar proporción de vecinos mayoritarios en k vecinos más cercanos
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        vecinos = nn.kneighbors(X_min, return_distance=False)[:, 1:]

        riesgo = []
        for idxs in vecinos:
            clases_vecinas = y[idxs]
            r = np.sum(clases_vecinas == 0) / self.k
            riesgo.append(r)
        riesgo = np.array(riesgo)

        # === Fase 2: Densidad local por intersección de áreas ===
        # Evalúa qué tan densa es la región local usando intersección de radios fijos
        vecinos_minor = NearestNeighbors(n_neighbors=self.k + 1).fit(X_min).kneighbors(X_min, return_distance=False)[:, 1:]
        densidades = self.calcular_densidad_interseccion(X_min, vecinos_minor, radio=self.radio_densidad)

        # === Filtro conjunto por riesgo y densidad ===
        if self.k == 5:
            r_mask = (riesgo >= 0.4) & (riesgo <= 0.6)
            percentil_corte = 75
        elif self.k == 7:
            r_mask = (riesgo >= 0.3) & (riesgo <= 0.7)
            percentil_corte = 70
        elif self.k == 9:
            r_mask = (riesgo >= 0.2) & (riesgo <= 0.8)
            percentil_corte = 65
        else:
            raise ValueError("Solo se admite k = 5, 7 o 9")

        # Eliminar muestras con densidad == 0 (zonas ruidosas)
        densidad_mask = densidades > 0.0
        combinacion_mask = r_mask & densidad_mask

        # Aplicar filtro a las muestras candidatas
        X_min_filtrado = X_min[combinacion_mask]
        riesgo_filtrado = riesgo[combinacion_mask]
        vecinos_filtrados = vecinos[combinacion_mask]

        # === Generación de muestras sintéticas ===
        n_sint = len(X_maj) - len(X_min)
        muestras_sinteticas = []

        for _ in range(n_sint):
            idx = self.random_state.randint(len(X_min_filtrado))
            xi = X_min_filtrado[idx]
            r_i = riesgo_filtrado[idx]
            idxs_vecinos = vecinos_filtrados[idx]

            # Selección adaptativa de vecinos válidos según un percentil de distancia
            distancias = np.linalg.norm(X[idxs_vecinos] - xi, axis=1)
            percentil = np.percentile(distancias, percentil_corte)
            vecinos_validos = idxs_vecinos[distancias <= percentil]

            if len(vecinos_validos) == 0:
                continue

            z_idx = self.random_state.choice(vecinos_validos)
            xz = X[z_idx]

            # δ adaptativo según el riesgo
            if 0.4 <= r_i < 0.5:
                delta = self.random_state.uniform(0.6, 0.8)
            elif 0.5 <= r_i <= 0.6:
                delta = self.random_state.uniform(0.3, 0.5)
            else:
                delta = self.random_state.uniform(0.4, 0.6)

            # Interpolación para crear el punto sintético
            xsint = xi + delta * (xz - xi)
            muestras_sinteticas.append(xsint)

        # Unir los nuevos datos con los originales
        X_sint = np.array(muestras_sinteticas)
        y_sint = np.ones(len(X_sint))

        X_resampled = np.vstack([X, X_sint])
        y_resampled = np.hstack([y, y_sint])
        return X_resampled, y_resampled
