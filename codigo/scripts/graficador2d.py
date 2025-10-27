# graficador_2d.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Iterable
import numpy as np
import matplotlib.pyplot as plt

try:
    import umap  # opcional
    _hay_umap = True
except Exception:
    _hay_umap = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import colors as mcolors


class Graficador2D:
    """
    Proyecta X (n x d) a 2D de forma CONSISTENTE y grafica:
      - Panel izquierdo: datos originales (X, y)
      - Panel derecho:   datos aumentados (X_res, y_res)

    Consistencia:
      • UNA sola base de proyección para ambos conjuntos.
      • Por defecto ajusta la proyección sobre X (fit_on="original").
      • Podés indicar fit_on="both" vía **kwargs_reductor para ajustar sobre [X; X_res].
    """

    def __init__(self,
                 reductor: str = "auto",
                 semilla: Optional[int] = None,
                 percentil_densidad: Optional[int] = None,
                 percentil_riesgo: Optional[int] = None,
                 criterio_pureza: Optional[str] = None,
                 nombre_dataset: Optional[str] = None,
                 **kwargs_reductor: Any) -> None:

        self.nombre_reductor = reductor.lower() if isinstance(reductor, str) else "auto"
        self.semilla = semilla

        # fit_on via kwargs para no romper llamadas existentes
        self.fit_on = kwargs_reductor.pop("fit_on", "original")  # "original" | "both"
        self.kwargs_reductor = kwargs_reductor

        # --- para el título ---
        self.percentil_densidad = percentil_densidad
        self.percentil_riesgo = percentil_riesgo
        self.criterio_pureza = criterio_pureza
        self.nombre_dataset = nombre_dataset

        # --- estado interno ---
        self._reductor = None            # PCA / UMAP / "TSNE_ESPECIAL" / None
        self._d_original: Optional[int] = None

    # ---------- Embedding / proyección ----------
    def _elegir_reductor(self, d: int):
        if self.nombre_reductor == "auto":
            if d <= 2:
                return None  # identidad
            return PCA(n_components=2, random_state=self.semilla)

        if self.nombre_reductor == "pca":
            return PCA(n_components=2, random_state=self.semilla)

        if self.nombre_reductor == "umap":
            if not _hay_umap:
                raise ImportError("UMAP no está disponible. Instala 'umap-learn' o usa reductor='pca'.")
            defaults = dict(n_components=2, n_neighbors=15, min_dist=0.1,
                            metric="euclidean", random_state=self.semilla)
            defaults.update(self.kwargs_reductor or {})
            return umap.UMAP(**defaults)

        if self.nombre_reductor == "tsne":
            # t-SNE no tiene transform() confiable → ruta especial (ajuste sobre X∪X_res).
            return "TSNE_ESPECIAL"

        raise ValueError(f"Reductor no soportado: {self.nombre_reductor}")

    def ajustar(self, X: np.ndarray) -> "Graficador2D":
        """Ajusta (fit) el reductor sobre X. Si d<=2 y 'auto', se usa identidad."""
        X = np.asarray(X)
        _, d = X.shape
        self._d_original = d

        if d <= 2 and (self.nombre_reductor in ("auto",) or self.nombre_reductor is None):
            self._reductor = None
            return self

        self._reductor = self._elegir_reductor(d)

        if self._reductor == "TSNE_ESPECIAL":
            # t-SNE se ajustará luego sobre X y X_res combinados (ver incrustar_par)
            return self

        self._reductor.fit(X)
        return self

    def transformar(self, X: np.ndarray) -> np.ndarray:
        """Transforma X a 2D usando el reductor ajustado."""
        X = np.asarray(X)

        if self._reductor is None:
            return X[:, 0:2] if X.shape[1] > 2 else X

        if self._reductor == "TSNE_ESPECIAL":
            raise RuntimeError("t-SNE no soporta transform() estable. Usá 'incrustar_par(X, X_res)'.")

        return self._reductor.transform(X)

    def ajustar_transformar(self, X: np.ndarray) -> np.ndarray:
        """Ajusta y transforma X en una sola llamada."""
        self.ajustar(X)
        return self.transformar(X)

    def incrustar_par(self, X: np.ndarray, X_res: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna (Z, Z_res) en 2D usando UNA base de proyección.
        - fit_on="original": ajusta con X y transforma X y X_res.
        - fit_on="both": ajusta con [X; X_res] (PCA/UMAP) y proyecta ambos en esa base.
        - t-SNE: ajusta siempre sobre [X; X_res] por limitaciones del método.
        """
        X = np.asarray(X); X_res = np.asarray(X_res)

        if self.nombre_reductor == "tsne":
            ambos = np.vstack([X, X_res])
            tsne = TSNE(n_components=2, random_state=self.semilla, **self.kwargs_reductor)
            ambos_2d = tsne.fit_transform(ambos)
            return ambos_2d[:len(X)], ambos_2d[len(X):]

        # PCA/UMAP/Identidad
        if self.fit_on == "both":
            X_fit = np.vstack([X, X_res])
            self.ajustar(X_fit)
        else:
            self.ajustar(X)

        Z = self.transformar(X)
        Z_res = self.transformar(X_res)
        return Z, Z_res

    # ---------- Utils de clases/leyenda/colores ----------
    @staticmethod
    def _unicos_en_orden(seq: Iterable) -> list:
        """Devuelve los elementos únicos preservando el primer orden de aparición."""
        vistos, out = set(), []
        for v in seq:
            if v not in vistos:
                vistos.add(v); out.append(v)
        return out

    @staticmethod
    def _nombre_de_clase(clase_id, nombres: Optional[Dict[Any, str] | list | tuple]) -> str:
        if nombres is None:
            return f"Clase {clase_id}"
        if isinstance(nombres, dict):
            return str(nombres.get(clase_id, f"Clase {clase_id}"))
        if isinstance(nombres, (list, tuple)):
            try:
                idx = int(clase_id)
                if 0 <= idx < len(nombres):
                    return str(nombres[idx])
            except Exception:
                pass
        return f"Clase {clase_id}"

    @classmethod
    def _etiqueta(cls, clase_id, cantidad: int, nombres: Optional[Dict[Any, str] | list | tuple]) -> str:
        return f"{cls._nombre_de_clase(clase_id, nombres)} (n={cantidad})"

    @staticmethod
    def _paleta_base(n: int) -> list[str]:
        base = list(plt.rcParams['axes.prop_cycle'].by_key().get('color', []))
        if not base:
            base = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        if n <= len(base):
            return base[:n]
        # Extender de forma determinista usando tab20 → hex
        tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
        extra_hex = [mcolors.to_hex(c) for c in tab20]
        ciclo = [extra_hex[i % len(extra_hex)] for i in range(n - len(base))]
        return base + ciclo

    @staticmethod
    def _construir_mapa_colores(clases_orden_global: list, paleta: Optional[list[str]] = None) -> dict:
        n = len(clases_orden_global)
        colores = paleta[:] if paleta else Graficador2D._paleta_base(n)
        if len(colores) < n:
            faltan = n - len(colores)
            colores += Graficador2D._paleta_base(faltan)
        return {c: colores[i] for i, c in enumerate(clases_orden_global)}

    # ---------- Graficación ----------
    def trazar_original_vs_aumentado(self,
                                     X: np.ndarray, y: np.ndarray,
                                     X_res: np.ndarray, y_res: np.ndarray,
                                     titulo: str = "Original vs. Aumentado",
                                     nombres_clase: Optional[Dict[Any, str] | list | tuple] = None,
                                     tam_punto: int = 22,
                                     alpha: float = 0.85,
                                     tam_fig: Tuple[int, int] = (12, 5),
                                     paleta: Optional[list[str]] = None) -> None:
        """
        Dibuja dos paneles usando la MISMA proyección y los MISMOS colores por clase.
        - La leyenda sigue el orden de aparición en 'y'; luego agrega las clases
          que aparezcan sólo en 'y_res'.
        - 'nombres_clase' puede ser dict o lista/tupla (índice = etiqueta).
        """
        X = np.asarray(X); y = np.asarray(y)
        X_res = np.asarray(X_res); y_res = np.asarray(y_res)

        # Proyección consistente (fit_on controla base de ajuste)
        Z, Z_res = self.incrustar_par(X, X_res)

        # Orden global de clases (estable para ambos paneles)
        orden_y = self._unicos_en_orden(y)
        orden_yres = [c for c in self._unicos_en_orden(y_res) if c not in set(orden_y)]
        clases_global = orden_y + orden_yres

        # Mapa clase→color estable
        color_map = self._construir_mapa_colores(clases_global, paleta=paleta)

        fig, axes = plt.subplots(1, 2, figsize=tam_fig, constrained_layout=True)
        ax1, ax2 = axes

        # Panel original
        for c in clases_global:
            m = (y == c)
            if not np.any(m):
                continue
            ax1.scatter(Z[m, 0], Z[m, 1], s=tam_punto, alpha=alpha,
                        color=color_map[c],
                        label=self._etiqueta(c, int(m.sum()), nombres_clase))
        ax1.set_title("Original")
        ax1.set_xlabel("Componente 1"); ax1.set_ylabel("Componente 2")
        ax1.legend(loc="best", frameon=True)

        # Panel aumentado
        for c in clases_global:
            m = (y_res == c)
            if not np.any(m):
                continue
            ax2.scatter(Z_res[m, 0], Z_res[m, 1], s=tam_punto, alpha=alpha,
                        color=color_map[c],
                        label=self._etiqueta(c, int(m.sum()), nombres_clase))
        ax2.set_title("Aumentado")
        ax2.set_xlabel("Componente 1"); ax2.set_ylabel("Componente 2")
        ax2.legend(loc="best", frameon=True)

        # Título informativo
        dens = getattr(self, "percentil_densidad", None)
        ries = getattr(self, "percentil_riesgo", None)
        pureza = getattr(self, "criterio_pureza", None)
        dataset = getattr(self, "nombre_dataset", "Dataset")

        linea_superior = f"{dataset} — Densidad: {dens} | Riesgo: {ries} | Pureza: {pureza}"
        linea_inferior = f"{titulo} (PCSMOTE)"
        fig.suptitle(f"{linea_superior}\n{linea_inferior}", fontsize=12, fontweight="bold")

        plt.show()
