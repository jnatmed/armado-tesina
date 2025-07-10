from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
import numpy as np

class PCSMOTE:
    def __init__(self, k_neighbors=5, random_state=None,
                 radio_densidad=1.0, percentil_dist=75,
                 percentil_entropia=None, percentil_densidad=None,
                 criterio_pureza='entropia'):  
        self.k = k_neighbors
        self.random_state = check_random_state(random_state)
        self.radio_densidad = radio_densidad
        self.percentil_dist = percentil_dist
        self.percentil_entropia = percentil_entropia
        self.percentil_densidad = percentil_densidad
        self.criterio_pureza = criterio_pureza  

    def calcular_densidad_interseccion(self, X_min, vecinos, radio):
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
        X = np.array(X)
        y = np.array(y)

        X_min = X[y == 1]
        X_maj = X[y == 0]

        print(f"📌 Total muestras minoritarias: {len(X_min)}")
        print(f"📌 Total muestras mayoritarias: {len(X_maj)}")

        # Riesgo (proporción de vecinos mayoritarios)
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        vecinos = nn.kneighbors(X_min, return_distance=False)[:, 1:]
        riesgo = np.array([np.sum(y[idxs] == 0) / self.k for idxs in vecinos])
        
        print(f"📊 Riesgo - media: {riesgo.mean():.4f} | min: {riesgo.min():.4f} | max: {riesgo.max():.4f}")

        # Densidad local por intersección
        vecinos_minor = NearestNeighbors(n_neighbors=self.k + 1).fit(X_min).kneighbors(X_min, return_distance=False)[:, 1:]
        densidades = self.calcular_densidad_interseccion(X_min, vecinos_minor, radio=self.radio_densidad)

        print(f"📊 Densidad - media: {densidades.mean():.4f} | p25: {np.percentile(densidades, 25):.4f} | p50: {np.percentile(densidades, 50):.4f} | p75: {np.percentile(densidades, 75):.4f}")

        # Filtros por riesgo y densidad
        r_mask = (riesgo >= 0.4) & (riesgo <= 0.6)
        densidad_mask = densidades > 0.0
        combinacion_mask = r_mask & densidad_mask

        print(f"🔎 Muestras con riesgo ∈ [0.4, 0.6]: {np.sum(r_mask)}")
        print(f"🔎 Muestras con densidad > 0: {np.sum(densidad_mask)}")
        print(f"✅ Muestras seleccionadas para sobremuestreo (intersección): {np.sum(combinacion_mask)}")
        print(f"❌ Muestras descartadas (ruido o baja densidad): {len(X_min) - np.sum(combinacion_mask)}")

        # Filtrado final
        X_min_filtrado = X_min[combinacion_mask]
        
        if len(X_min_filtrado) == 0:
            print("⚠️ No se encontraron muestras válidas para sobremuestreo. Devolviendo conjunto original.")
            return X.copy(), y.copy()

        riesgo_filtrado = riesgo[combinacion_mask]
        vecinos_filtrados = vecinos[combinacion_mask]

        # Cantidad de muestras sintéticas a generar
        n_sint = len(X_maj) - len(X_min)
        print(f"🔁 Muestras sintéticas a generar: {n_sint}")

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

        print(f"✅ Muestras sintéticas generadas efectivamente: {len(X_sint)}")

        X_resampled = np.vstack([X, X_sint])
        y_resampled = np.hstack([y, y_sint])
        return X_resampled, y_resampled
