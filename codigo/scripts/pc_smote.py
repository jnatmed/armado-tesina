from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from scipy.stats import entropy
import numpy as np
from collections import Counter

class PCSMOTE:
    def __init__(self, k_neighbors=5, random_state=None,
                 radio_densidad=1.0, percentil_dist=75,
                 percentil_entropia=None, percentil_densidad=None,
                 criterio_pureza='entropia', modo_espacial='2d',
                 factor_equilibrio=0.8, verbose=True):
        self.k = k_neighbors
        self.random_state = check_random_state(random_state)
        self.radio_densidad = radio_densidad
        self.percentil_dist = percentil_dist
        self.percentil_entropia = percentil_entropia
        self.percentil_densidad = percentil_densidad
        self.criterio_pureza = criterio_pureza
        self.modo_espacial = modo_espacial
        self.factor_equilibrio = factor_equilibrio
        self.verbose = verbose

    def calcular_densidad_interseccion(self, X_min, vecinos, radio):
        densidades = []
        for i, xi in enumerate(X_min):
            intersecciones = 0
            for j in vecinos[i]:
                xj = X_min[j]
                distancia = np.linalg.norm(xi[:3] - xj[:3]) if self.modo_espacial == '3d' else np.linalg.norm(xi - xj)
                if distancia <= 2 * radio:
                    intersecciones += 1
            densidades.append(intersecciones / len(vecinos[i]))
        return np.array(densidades)

    def calcular_entropia(self, vecinos, y):
        entropias = []
        for idxs in vecinos:
            clases, counts = np.unique(y[idxs], return_counts=True)
            p = counts / counts.sum()
            entropias.append(entropy(p, base=2))
        return np.array(entropias)

    def fit_resample(self, X, y, max_sinteticas=None):
        X = np.array(X)
        y = np.array(y)

        X_min = X[y == 1]
        X_maj = X[y == 0]

        if self.verbose:
            print(f"📌 Total muestras minoritarias: {len(X_min)}")
            print(f"📌 Total muestras mayoritarias: {len(X_maj)}")

        if len(X_min) < self.k + 1:
            if self.verbose:
                print(f"⚠️ Muy pocas muestras minoritarias ({len(X_min)}). Se requieren al menos {self.k + 1}. Devolviendo dataset original.")
            return X.copy(), y.copy()

        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        vecinos = nn.kneighbors(X_min, return_distance=False)[:, 1:]
        riesgo = np.array([np.sum(y[idxs] == 0) / self.k for idxs in vecinos])

        vecinos_minor = NearestNeighbors(n_neighbors=self.k + 1).fit(X_min).kneighbors(X_min, return_distance=False)[:, 1:]
        densidades = self.calcular_densidad_interseccion(X_min, vecinos_minor, self.radio_densidad)

        if self.criterio_pureza == 'entropia':
            entropias = self.calcular_entropia(vecinos, y)
            pureza_mask = entropias <= (np.percentile(entropias, self.percentil_entropia)
                                        if self.percentil_entropia is not None else 1.0)
        elif self.criterio_pureza == 'proporcion':
            proporciones_min = np.array([np.sum(y[idxs] == 1) / self.k for idxs in vecinos])
            pureza_mask = (proporciones_min >= 0.4) & (proporciones_min <= 0.6)
        else:
            raise ValueError(f"Criterio de pureza no reconocido: {self.criterio_pureza}")

        if self.percentil_densidad is not None:
            umbral_densidad = np.percentile(densidades, self.percentil_densidad)
            densidad_mask = densidades >= umbral_densidad
        else:
            densidad_mask = densidades > 0.0

        combinacion_mask = pureza_mask & densidad_mask
        X_min_filtrado = X_min[combinacion_mask]

        if len(X_min_filtrado) < self.k + 1:
            if self.verbose:
                print(f"⚠️ Muy pocas muestras luego del filtrado ({len(X_min_filtrado)}). Devolviendo dataset original.")
            return X.copy(), y.copy()

        riesgo_filtrado = riesgo[combinacion_mask]
        vecinos_filtrados = vecinos[combinacion_mask]

        n_sint = max_sinteticas if max_sinteticas is not None else len(X_maj) - len(X_min)
        muestras_sinteticas = []

        for _ in range(n_sint):
            idx = self.random_state.randint(len(X_min_filtrado))
            xi = X_min_filtrado[idx]
            r_i = riesgo_filtrado[idx]
            idxs_vecinos = vecinos_filtrados[idx]

            distancias = (np.linalg.norm(X[idxs_vecinos][:, :3] - xi[:3], axis=1)
                          if self.modo_espacial == '3d'
                          else np.linalg.norm(X[idxs_vecinos] - xi, axis=1))
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

            muestras_sinteticas.append(xi + delta * (xz - xi))

        X_sint = np.array(muestras_sinteticas)
        y_sint = np.ones(len(X_sint))

        if self.verbose:
            print(f"🧬 Se generaron {len(X_sint)} muestras sintéticas para la clase minoritaria (y=1)")
            if max_sinteticas is not None:
                print(f"📈 Objetivo de generación: {max_sinteticas} muestras")
                if len(X_sint) < max_sinteticas:
                    print(f"⚠️ Solo se generaron {len(X_sint)} de {max_sinteticas} posibles (limitado por filtrado o vecinos)")

        X_resampled = np.vstack([X, X_sint])
        y_resampled = np.hstack([y, y_sint])
        return X_resampled, y_resampled

    def fit_resample_multiclass(self, X, y):
        clases = np.unique(y)
        X_res = X.copy()
        y_res = y.copy()
        conteo_original = Counter(y)
        max_count = max(conteo_original.values())

        if self.verbose:
            print("📊 Distribución de clases en el conjunto de entrenamiento (antes del sobremuestreo):")
            print("╔══════════╦════════════╦════════════════════╦════════════════════════════════════╗")
            print("║  Clase   ║   Train    ║ Objetivo (balance) ║   Estado                            ║")
            print("╠══════════╬════════════╬════════════════════╬════════════════════════════════════╣")
            for clase in sorted(conteo_original):
                original = conteo_original[clase]
                objetivo = int(max_count * self.factor_equilibrio)

                if original >= objetivo:
                    estado = "✅ No se sobremuestrea (ya cumple o excede)"
                else:
                    estado = "⬆ Será sobremuestreada"

                print(f"║   {str(clase):<6}  ║   {original:<8} ║   {objetivo:<16} ║   {estado:<36} ║")
            print("╚══════════╩════════════╩════════════════════╩════════════════════════════════════╝")

        conteo_sinteticas = Counter()

        for clase in clases:
            y_bin = (y == clase).astype(int)
            actual = np.sum(y_bin)
            objetivo = int(max_count * self.factor_equilibrio)

            if self.verbose:
                print(f"⚖️ [Clase {clase}] Aplicando factor_equilibrio={self.factor_equilibrio} → objetivo: {objetivo} muestras")

            cantidad_faltante = max(0, objetivo - actual)

            if cantidad_faltante > 0:
                sampler_tmp = PCSMOTE(
                    k_neighbors=self.k,
                    random_state=self.random_state,
                    radio_densidad=self.radio_densidad,
                    percentil_dist=self.percentil_dist,
                    percentil_entropia=self.percentil_entropia,
                    percentil_densidad=self.percentil_densidad,
                    criterio_pureza=self.criterio_pureza,
                    modo_espacial=self.modo_espacial,
                    factor_equilibrio=self.factor_equilibrio,
                    verbose=False
                )
                X_bin_res, y_bin_res = sampler_tmp.fit_resample(X, y_bin, max_sinteticas=cantidad_faltante)
                nuevos = len(X_bin_res) - len(X)
                if nuevos > 0:
                    X_nuevos = X_bin_res[-nuevos:]
                    y_nuevos = np.full(nuevos, clase)
                    X_res = np.vstack([X_res, X_nuevos])
                    y_res = np.hstack([y_res, y_nuevos])
                    conteo_sinteticas[clase] += nuevos
                    if self.verbose:
                        print(f"🧬 Clase {clase}: {nuevos} muestras sintéticas generadas")
                        if nuevos > cantidad_faltante:
                            print(f"⚠️ Advertencia: Clase {clase} generó más muestras ({nuevos}) que las esperadas ({cantidad_faltante})")
                        elif nuevos < cantidad_faltante:
                            print(f"⚠️ Solo se generaron {nuevos} de {cantidad_faltante} esperadas (limitado por filtrado o vecinos)")

        if self.verbose and conteo_sinteticas:
            print("📊 Total de muestras sintéticas generadas por clase:")
            for clase, cantidad in conteo_sinteticas.items():
                print(f"   🧪 Clase {clase}: {cantidad} muestras")

        return X_res, y_res
