import numpy as np
from alfa_dbasmote import AlphaDistanceDBASMOTE
from ar_adasyn import ARADASYN
from pc_smote import PCSMOTE
from sklearn.base import BaseEstimator

class AlphaDistanceDBASMOTEWrapper:
    def __init__(self, beta=1.0, m=5, random_state=None):
        self.beta = beta
        self.m = m
        self.random_state = random_state

    def fit_resample(self, X_min, X_maj):
        X_syn = AlphaDistanceDBASMOTE(X_min, X_maj, beta=self.beta, m=self.m, random_state=self.random_state)
        X_resampled = np.vstack([X_min, X_maj, X_syn])
        y_resampled = np.array([1]*len(X_min) + [0]*len(X_maj) + [1]*len(X_syn))
        return X_resampled, y_resampled


class ARADASYNWrapper:
    def __init__(self, k=5, random_state=None):
        self.k = k
        self.random_state = random_state

    def fit_resample(self, X_min, X_maj):
        X_syn = ARADASYN(X_min, X_maj, k=self.k, random_state=self.random_state)
        X_resampled = np.vstack([X_min, X_maj, X_syn])
        y_resampled = np.array([1]*len(X_min) + [0]*len(X_maj) + [1]*len(X_syn))
        return X_resampled, y_resampled

class PCSMOTEWrapper(BaseEstimator):
    def __init__(
        self,
        # Hiperparámetros que vas a tunear desde RandomizedSearchCV:
        percentil_densidad=50,
        percentil_riesgo=50,          # alias que mapeamos a percentil_dist de PCSMOTE
        criterio_pureza="entropia",
        # Otros parámetros típicos del sampler:
        k_neighbors=5,
        random_state=42,
        percentil_entropia=75,        # solo aplica si criterio_pureza == "entropia"
        modo_espacial="2d",
        factor_equilibrio=0.8,
        # opcionales según tu implementación interna:
        radio_densidad=1.0,
        verbose=False,
        max_total_multiplier=None,
        max_sinteticas_por_clase=None
    ):
        # Guardamos todos los params como atributos de instancia.
        # (BaseEstimator usará esto para get_params/set_params)
        self.percentil_densidad = percentil_densidad
        self.percentil_riesgo = percentil_riesgo
        self.criterio_pureza = criterio_pureza
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.percentil_entropia = percentil_entropia
        self.modo_espacial = modo_espacial
        self.factor_equilibrio = factor_equilibrio
        self.radio_densidad = radio_densidad
        self.verbose = verbose
        self.max_total_multiplier = max_total_multiplier
        self.max_sinteticas_por_clase = max_sinteticas_por_clase

    def fit(self, X, y):
        # Requisito de sklearn; no aprendemos estado persistente acá.
        return self

    def fit_resample(self, X, y):
        # Mapeo de alias: tu PCSMOTE usa percentil_dist internamente.
        percentil_dist = self.percentil_riesgo

        sampler = PCSMOTE(
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
            radio_densidad=self.radio_densidad,
            percentil_densidad=self.percentil_densidad,
            percentil_dist=percentil_dist,  # <- mapeo clave
            percentil_entropia=(
                self.percentil_entropia if self.criterio_pureza == "entropia" else None
            ),
            criterio_pureza=self.criterio_pureza,
            modo_espacial=self.modo_espacial,
            factor_equilibrio=self.factor_equilibrio,
            verbose=self.verbose,
            max_total_multiplier=self.max_total_multiplier,
            max_sinteticas_por_clase=self.max_sinteticas_por_clase
        )

        # Binario vs multiclase (usa método multiclase si existe y corresponde)
        try:
            from sklearn.utils.multiclass import type_of_target
            es_multiclase = ('multiclass' in type_of_target(y))
        except Exception:
            es_multiclase = False

        if es_multiclase and hasattr(sampler, "fit_resample_multiclass"):
            return sampler.fit_resample_multiclass(X, y)
        return sampler.fit_resample(X, y)