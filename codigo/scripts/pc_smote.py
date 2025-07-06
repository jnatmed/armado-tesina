from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
import numpy as np

class PCSMOTE:
    def __init__(self, sampling_strategy='auto', k_neighbors=5, random_state=None):
        self.sampling_strategy = sampling_strategy
        self.k = k_neighbors
        self.random_state = check_random_state(random_state)

    def fit_resample(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Separar clases
        X_min = X[y == 1]
        X_maj = X[y == 0]

        # Calcular riesgo r_i = #mayoritarios / k
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        vecinos = nn.kneighbors(X_min, return_distance=False)[:, 1:]
        riesgo = []
        for idxs in vecinos:
            clases_vecinas = y[idxs]
            r = np.sum(clases_vecinas == 0) / self.k
            riesgo.append(r)

        riesgo = np.array(riesgo)
        # Filtrar por rango de riesgo según k
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

        X_min_filtrado = X_min[r_mask]
        riesgo_filtrado = riesgo[r_mask]
        vecinos_filtrados = vecinos[r_mask]

        # Cantidad de muestras a generar
        n_sint = len(X_maj) - len(X_min)
        muestras_sinteticas = []

        for _ in range(n_sint):
            idx = self.random_state.randint(len(X_min_filtrado))
            xi = X_min_filtrado[idx]
            r_i = riesgo_filtrado[idx]
            idxs_vecinos = vecinos_filtrados[idx]

            # Vecinos minoritarios (con distancia filtrada por percentil)
            distancias = np.linalg.norm(X[idxs_vecinos] - xi, axis=1)
            percentil = np.percentile(distancias, percentil_corte)
            vecinos_validos = idxs_vecinos[distancias <= percentil]

            if len(vecinos_validos) == 0:
                continue

            z_idx = self.random_state.choice(vecinos_validos)
            xz = X[z_idx]

            # Rango adaptativo de δ según r_i
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
