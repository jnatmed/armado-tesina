# graficador_2d.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Iterable
import numpy as np
import matplotlib.pyplot as plt
import itertools

try:
    import umap  # opcional
    _hay_umap = True
except Exception:
    _hay_umap = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class Graficador2D:
    """
    Proyecta X (n x d) a 2D de forma CONSISTENTE y grafica:
      - Panel izquierdo: datos originales (X, y)
      - Panel derecho:   datos aumentados (X_res, y_res)

    Conservar colores por clase:
      • Se construye un mapa clase→color usando el conjunto global de clases (y ∪ y_res).
      • La leyenda se ordena por el orden de aparición en y, y luego se agregan las
        que sólo aparezcan en y_res.
    """

    def __init__(self,
                 reductor: str = "auto",
                 escalar: bool = True,
                 semilla: Optional[int] = None,
                 **kwargs_reductor: Any) -> None:
        self.nombre_reductor = reductor.lower()
        self.escalar = bool(escalar)
        self.semilla = semilla
        self.kwargs_reductor = kwargs_reductor

        self._escalador: Optional[StandardScaler] = None
        self._reductor = None  # PCA/UMAP/"TSNE_ESPECIAL"/None
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

    def _ajustar_posible_escalado(self, X: np.ndarray) -> np.ndarray:
        if not self.escalar:
            return X
        self._escalador = StandardScaler()
        return self._escalador.fit_transform(X)

    def _transformar_posible_escalado(self, X: np.ndarray) -> np.ndarray:
        if not self.escalar or self._escalador is None:
            return X
        return self._escalador.transform(X)

    def ajustar(self, X: np.ndarray) -> "Graficador2D":
        """Ajusta (fit) el reductor sobre X. Si d<=2 y 'auto', se usa identidad."""
        X = np.asarray(X)
        _, d = X.shape
        self._d_original = d

        if d <= 2 and (self.nombre_reductor in ("auto",) or self.nombre_reductor is None):
            self._reductor = None
            self._escalador = StandardScaler() if self.escalar else None
            if self._escalador is not None:
                self._escalador.fit(X)
            return self

        self._reductor = self._elegir_reductor(d)

        if self._reductor == "TSNE_ESPECIAL":
            # t-SNE se ajustará luego sobre X y X_res combinados (ver incrustar_par)
            self._ajustar_posible_escalado(X)
            return self

        Xs = self._ajustar_posible_escalado(X)
        self._reductor.fit(Xs)
        return self

    def transformar(self, X: np.ndarray) -> np.ndarray:
        """Transforma X a 2D usando el reductor ajustado."""
        X = np.asarray(X)

        if self._reductor is None:
            return X[:, 0:2] if X.shape[1] > 2 else X

        if self._reductor == "TSNE_ESPECIAL":
            raise RuntimeError("t-SNE no soporta transform() estable. Usá 'incrustar_par(X, X_res)'.")

        Xs = self._transformar_posible_escalado(X)
        return self._reductor.transform(Xs)

    def ajustar_transformar(self, X: np.ndarray) -> np.ndarray:
        """Ajusta y transforma X en una sola llamada."""
        self.ajustar(X)
        return self.transformar(X)

    def incrustar_par(self, X: np.ndarray, X_res: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (Z, Z_res) en 2D con proyección CONSISTENTE.
        • PCA/UMAP/Identidad: fit en X, transform en X y X_res.
        • t-SNE: fit sobre concat([X, X_res]) y luego separa.
        """
        X = np.asarray(X); X_res = np.asarray(X_res)
        if self._reductor == "TSNE_ESPECIAL":
            Xs = self._ajustar_posible_escalado(X)
            Xs_res = self._transformar_posible_escalado(X_res)
            ambos = np.vstack([Xs, Xs_res])

            tsne = TSNE(n_components=2, random_state=self.semilla, **self.kwargs_reductor)
            ambos_2d = tsne.fit_transform(ambos)
            Z, Z_res = ambos_2d[:len(X)], ambos_2d[len(X):]
            return Z, Z_res

        # PCA/UMAP/Identidad
        self.ajustar(X)
        Z = self.transformar(X)
        Z_res = self.transformar(X_res)
        return Z, Z_res

    # ---------- Utils de clases/leyenda/colores ----------
    @staticmethod
    def _unicos_en_orden(seq: Iterable) -> list:
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
        # extender de forma determinista
        extra = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()
        extra_hex = [plt.colors.to_hex(c) if hasattr(plt, "colors") else None for c in extra]
        extra_hex = [c if c is not None else base[i % len(base)] for i, c in enumerate(extra_hex)]
        ciclo = list(itertools.islice(itertools.cycle(extra_hex), n - len(base)))
        return base + ciclo

    @staticmethod
    def _construir_mapa_colores(clases_orden_global: list, paleta: Optional[list[str]] = None) -> dict:
        n = len(clases_orden_global)
        colores = paleta[:] if paleta else Graficador2D._paleta_base(n)
        if len(colores) < n:
            # extender si la paleta provista es corta
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

        # Proyección consistente
        Z, Z_res = self.incrustar_par(X, X_res)

        # Orden global de clases (estable para ambos paneles)
        orden_y = self._unicos_en_orden(y)
        orden_yres = self._unicos_en_orden([c for c in y_res if c not in set(orden_y)])
        clases_global = orden_y + orden_yres

        # Mapa clase→color estable
        color_map = self._construir_mapa_colores(clases_global, paleta=paleta)

        fig, axes = plt.subplots(1, 2, figsize=tam_fig, constrained_layout=True)
        ax1, ax2 = axes

        # Panel original (usar SIEMPRE el mismo orden para que las leyendas coincidan)
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

        fig.suptitle(titulo)
        plt.show()
