from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
import numpy as np

class PCSMOTE:
    def __init__(self, k_neighbors=5, random_state=None,
                 radio_densidad=1.0, percentil_dist=75,
                 percentil_entropia=None, percentil_densidad=None):
        """
        Implementación de PC-SMOTE (Percentile-Controlled SMOTE) con criterios de:
        - riesgo basado en vecinos de clases diferentes
        - densidad basada en intersección geométrica local
        - selección adaptativa de vecinos basada en percentiles

        Parámetros:
        - k_neighbors: cantidad de vecinos a considerar
        - random_state: estado aleatorio para reproducibilidad
        - radio_densidad: radio fijo para estimar densidad por intersección
        - percentil_dist: percentil usado para selección de vecinos válidos (interpolación)
        - percentil_entropia: umbral para pureza (no implementado aún)
        - percentil_densidad: umbral para densidad (no implementado aún)
        """
        self.k = k_neighbors
        self.random_state = check_random_state(random_state)
        self.radio_densidad = radio_densidad
        self.percentil_dist = percentil_dist
        self.percentil_entropia = percentil_entropia
        self.percentil_densidad = percentil_densidad

    def calcular_densidad_interseccion(self, X_min, vecinos, radio):
        """
        Calcula la densidad como cantidad de intersecciones circulares sobre k vecinos.
        Cada punto define un círculo de radio fijo; se cuenta cuántos se superponen.
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
        Ejecuta PC-SMOTE con:
        1. Estimación de riesgo (proporción de vecinos mayoritarios)
        2. Cálculo de densidad local (por intersección)
        3. Filtro conjunto y generación sintética adaptativa

        Entrada:
        - X: matriz de atributos (numpy array)
        - y: vector de etiquetas binarias (0=mayoritaria, 1=minoritaria)

        Salida:
        - X_resampled: matriz extendida con instancias sintéticas
        - y_resampled: etiquetas correspondientes
        """
        X = np.array(X)
        y = np.array(y)

        X_min = X[y == 1]
        X_maj = X[y == 0]

        # Riesgo: vecinos mixtos
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        vecinos = nn.kneighbors(X_min, return_distance=False)[:, 1:]
        riesgo = [np.sum(y[idxs] == 0) / self.k for idxs in vecinos]
        riesgo = np.array(riesgo)

        # Densidad local
        vecinos_minor = NearestNeighbors(n_neighbors=self.k + 1).fit(X_min).kneighbors(X_min, return_distance=False)[:, 1:]
        densidades = self.calcular_densidad_interseccion(X_min, vecinos_minor, radio=self.radio_densidad)

        # Filtro por riesgo y densidad mínima
        r_mask = (riesgo >= 0.4) & (riesgo <= 0.6)
        densidad_mask = densidades > 0.0
        combinacion_mask = r_mask & densidad_mask

        X_min_filtrado = X_min[combinacion_mask]
        riesgo_filtrado = riesgo[combinacion_mask]
        vecinos_filtrados = vecinos[combinacion_mask]

        n_sint = len(X_maj) - len(X_min)
        muestras_sinteticas = []

        for _ in range(n_sint):
            idx = self.random_state.randint(len(X_min_filtrado))
            xi = X_min_filtrado[idx]
            r_i = riesgo_filtrado[idx]
            idxs_vecinos = vecinos_filtrados[idx]

            distancias = np.linalg.norm(X[idxs_vecinos] - xi, axis=1)
            umbral = np.percentile(distancias, self.percentil_dist)
            vecinos_validos = idxs_vecinos[distancias <= umbral]

            if len(vecinos_validos) == 0:
                continue

            z_idx = self.random_state.choice(vecinos_validos)
            xz = X[z_idx]

            # δ adaptativo según riesgo
            if 0.4 <= r_i < 0.5:
                delta = self.random_state.uniform(0.6, 0.8)
            elif 0.5 <= r_i <= 0.6:
                delta = self.random_state.uniform(0.3, 0.5)
            else:
                delta = self.random_state.uniform(0.4, 0.6)

            xsint = xi + delta * (xz - xi)
            muestras_sinteticas.append(xsint)

        X_sint = np.array(muestras_sinteticas)
        y_sint = np.ones(len(X_sint))

        X_resampled = np.vstack([X, X_sint])
        y_resampled = np.hstack([y, y_sint])
        return X_resampled, y_resampled
