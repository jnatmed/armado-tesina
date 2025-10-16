# scripts/gestor_cache.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from cache import PCSMOTECache


class PCSMOTEGestorCache:
    """
    Gestor intermedio entre PCSMOTE (algoritmo) y PCSMOTECache (persistencia).

    Administra los cómputos costosos reutilizables:
        • Sigmas (auto-escalado LSD): sigma_X y sigma_Xmin
        • Vecindarios globales y locales (por LSD)
        • Umbrales LSD (locales por semilla y global)

    Estrategia:
      - Intenta cargar resultados desde caché (clave extendida con fingerprint de la minoritaria)
      - Si el caché es válido, reconstruye umbrales y retorna
      - Si no, calcula todo y guarda en caché

    Requiere un adaptador (típicamente la instancia del algoritmo) con:
        _compute_sigmas(X, k_sigma)
        _dists_lsd_seed(xi, X_ref, sigma_i, sigmas_ref)
        calcularUmbralDensidades(X_min, vecinos_min_local, percentil, k_sigma)

    Notas:
      - La clave del caché distingue la partición minoritaria actual vía:
            extra_meta = { "n_min": len(idxs_min_global), "pos_fp": sha1(idxs_min_global) }
      - Se valida que vecinos_min_local no exceda len(X_min) para evitar caché stale.
    """

    def __init__(self, cache, k: int, metrica_vecindario: str, percentil_dist: float):
        self.cache = cache
        self.k = int(k)
        self.metrica = str(metrica_vecindario)
        self.percentil_dist = float(percentil_dist)

    def obtener(
        self, X: np.ndarray, y: np.ndarray, nombre_dataset: str, adaptador
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna:
          vecinos_all_global : (n_min, k)  índices de vecinos (en X) para riesgo/pureza
          vecinos_min_local  : (n_min, k)  índices de vecinos minoritarios (en X_min)
          vecinos_min_global : (n_min, k)  mapeo de vecinos_min_local a índices globales (en X)
          sigma_X            : (n, )       sigma por punto en X
          sigma_Xmin         : (n_min, )   sigma por punto en X_min
        Efectos colaterales en 'adaptador':
          - adaptador._umbral_lsd_by_i
          - adaptador.umbral_distancia
        """
        # ----------------------- minoritaria + fingerprint -----------------------
        idxs_min_global = np.where(y == 1)[0]
        X_min = X[idxs_min_global]
        extra_meta = {
            "n_min": int(len(idxs_min_global)),
            "pos_fp": PCSMOTECache.fp_array(idxs_min_global),
        }

        # ----------------------------- intento de carga --------------------------
        datos_cache = None
        if self.cache is not None:
            datos_cache = self.cache.load(
                X, dataset=nombre_dataset, k=self.k, metric=self.metrica, extra=extra_meta
            )

        # -------------------------- usar caché si es válido ----------------------
        if datos_cache is not None:
            vmin = datos_cache["vecinos_min_local"].astype(int, copy=False)
            # sanity check: índices dentro del rango de X_min
            if vmin.size == 0 or vmin.max(initial=-1) >= len(X_min):
                datos_cache = None  # invalidar y pasar a recomputo

        if datos_cache is not None:
            adaptador._sigma_X    = datos_cache["sigma_X"].astype(float, copy=False)
            adaptador._sigma_Xmin = datos_cache["sigma_Xmin"].astype(float, copy=False)
            vecinos_all_global = datos_cache["vecinos_all_global"].astype(int, copy=False)
            vecinos_min_local  = datos_cache["vecinos_min_local"].astype(int, copy=False)
            vecinos_min_global = idxs_min_global[vecinos_min_local.astype(int)]

            # Reconstruir umbrales LSD (locales por semilla y global)
            adaptador._umbral_lsd_by_i = np.full(len(X_min), np.nan, dtype=float)
            todas_lsd = []
            for i in range(len(X_min)):
                idx_vecinos = vecinos_min_local[i]
                if idx_vecinos.size == 0:
                    continue
                dist_lsd = adaptador._dists_lsd_seed(
                    X_min[i],
                    X_min[idx_vecinos],
                    float(adaptador._sigma_Xmin[i]),
                    adaptador._sigma_Xmin[idx_vecinos],
                )
                if dist_lsd.size:
                    todas_lsd.append(dist_lsd)
                    adaptador._umbral_lsd_by_i[i] = float(
                        np.percentile(dist_lsd, self.percentil_dist)
                    )
            adaptador.umbral_distancia = (
                float(np.percentile(np.concatenate(todas_lsd), self.percentil_dist))
                if todas_lsd else 0.0
            )

            return (
                vecinos_all_global,
                vecinos_min_local,
                vecinos_min_global,
                adaptador._sigma_X,
                adaptador._sigma_Xmin,
            )

        # ------------------------------ recomputo -------------------------------
        nn_min_pre = NearestNeighbors(n_neighbors=self.k + 1).fit(X_min)
        vecinos_min_local_pre = nn_min_pre.kneighbors(X_min, return_distance=False)[:, 1:]

        adaptador._sigma_X    = adaptador._compute_sigmas(X,     k_sigma=max(1, min(self.k, len(X) - 1)))
        adaptador._sigma_Xmin = adaptador._compute_sigmas(X_min, k_sigma=self.k)

        adaptador.calcularUmbralDensidades(
            X_min=X_min,
            vecinos_min_local=vecinos_min_local_pre,
            percentil=self.percentil_dist,
            k_sigma=self.k,
        )

        vecinos_all_global = np.empty((len(X_min), self.k), dtype=int)
        vecinos_min_local  = np.empty((len(X_min), self.k), dtype=int)
        for i, xi in enumerate(X_min):
            dist_all = adaptador._dists_lsd_seed(xi, X,     adaptador._sigma_Xmin[i], adaptador._sigma_X)
            dist_min = adaptador._dists_lsd_seed(xi, X_min, adaptador._sigma_Xmin[i], adaptador._sigma_Xmin)
            dist_all[idxs_min_global[i]] = np.inf  # excluir self en global
            dist_min[i] = np.inf                   # excluir self en minoritaria
            vecinos_all_global[i] = np.argpartition(dist_all, self.k)[: self.k]
            vecinos_min_local[i]  = np.argpartition(dist_min, self.k)[: self.k]

        vecinos_min_global = idxs_min_global[vecinos_min_local.astype(int)]

        if self.cache is not None:
            self.cache.save(
                X,
                dataset=nombre_dataset,
                k=self.k,
                metric=self.metrica,
                sigma_X=adaptador._sigma_X,
                sigma_Xmin=adaptador._sigma_Xmin,
                vecinos_all_global=vecinos_all_global,
                vecinos_min_local=vecinos_min_local,
                extra_meta=extra_meta,  # distingue la minoritaria actual
            )

        return (
            vecinos_all_global,
            vecinos_min_local,
            vecinos_min_global,
            adaptador._sigma_X,
            adaptador._sigma_Xmin,
        )
