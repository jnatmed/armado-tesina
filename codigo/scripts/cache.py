# cache.py
from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


class PCSMOTECache:
    """
    Capa de persistencia en disco para reusar cómputos entre corridas
    del mismo dataset (vecindarios y sigmas, y opcionalmente 'candidatos_*').

    Guarda/carga: sigma_X, sigma_Xmin, vecinos_all_global, vecinos_min_local
    (y opcional: candidatos_all, candidatos_min).

    Clave: (dataset, shape, k, metric, fingerprint).
    Formato: .npz comprimido + meta JSON.
    """
    def __init__(self,
                 cache_dir=os.path.join(os.path.dirname(__file__), "cache"),
                 read: bool = True,
                 write: bool = True,
                 version: int = 1):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.read = bool(read)
        self.write = bool(write)
        self.version = int(version)

    # ---------- helpers de clave y path ----------
    # cache.py (añadir helper)
    @staticmethod
    def fp_array(arr: np.ndarray, max_elems: int = 4096) -> str:
        arr = np.asarray(arr, dtype=np.int64).ravel()
        n = min(max_elems, arr.size)
        import hashlib
        return hashlib.sha1(arr[:n].tobytes()).hexdigest()[:12]    
    
    def _fingerprint(self, X: np.ndarray, max_elems: int = 8192) -> str:
        """SHA1 de un preview de X (hasta 8k elementos) para evitar hashear todo."""
        X = np.asarray(X)
        flat = X.ravel()
        n = min(max_elems, flat.size)
        return hashlib.sha1(flat[:n].tobytes()).hexdigest()[:12]

    def make_key(self,
                 X: np.ndarray,
                 dataset: str,
                 k: int,
                 metric: str,
                 extra: Optional[Dict[str, Any]] = None) -> str:
        shape = f"{X.shape[0]}x{X.shape[1]}"
        fp = self._fingerprint(X)
        parts = [dataset or "unknown", shape, f"k{k}", str(metric), f"fp{fp}"]
        if extra:
            # Ordenar claves para estabilidad
            for kx in sorted(extra.keys()):
                parts.append(f"{kx}={extra[kx]}")
        return "__".join(parts)

    def path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    # ---------- API pública ----------
    def load(self, X, dataset, k, metric, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not self.read:
            return None
        key = self.make_key(X, dataset, k, metric, extra)  # << usa extra
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            meta = json.loads(data["meta"].tobytes().decode("utf-8"))

            if tuple(meta.get("shape", ())) != tuple(X.shape):
                return None
            if int(meta.get("k", -1)) != int(k):
                return None
            if str(meta.get("metric_vecindario", "")) != str(metric):
                return None
            if int(meta.get("version", -1)) != int(self.version):
                return None

            # Validación de ‘extra’ si se proveyó (robustez adicional)
            if extra is not None:
                for field in ("n_min", "pos_fp"):
                    if field in extra and meta.get(field) != extra[field]:
                        return None

            out = {
                "sigma_X": data["sigma_X"],
                "sigma_Xmin": data["sigma_Xmin"],
                "vecinos_all_global": data["vecinos_all_global"],
                "vecinos_min_local": data["vecinos_min_local"],
                "meta": meta,
            }
            if "candidatos_all" in data.files:
                out["candidatos_all"] = data["candidatos_all"]
            if "candidatos_min" in data.files:
                out["candidatos_min"] = data["candidatos_min"]
            return out
        except Exception:
            return None


    def save(self,
             X: np.ndarray,
             dataset: str,
             k: int,
             metric: str,
             sigma_X: np.ndarray,
             sigma_Xmin: np.ndarray,
             vecinos_all_global: np.ndarray,
             vecinos_min_local: np.ndarray,
             candidatos_all: Optional[np.ndarray] = None,
             candidatos_min: Optional[np.ndarray] = None,
             extra_meta: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        if not self.write:
            return None
        key = self.make_key(X, dataset, k, metric, extra=extra_meta)
        path = self.path_for(key)
        try:
            meta = {
                "dataset": dataset or "unknown",
                "shape": X.shape,
                "k": int(k),
                "metric_vecindario": str(metric),
                "version": int(self.version),
            }
            if extra_meta:
                meta.update(extra_meta)

            # Construir kwargs de arrays a guardar
            arrays = dict(
                sigma_X=np.asarray(sigma_X),
                sigma_Xmin=np.asarray(sigma_Xmin),
                vecinos_all_global=np.asarray(vecinos_all_global, dtype=np.int64),
                vecinos_min_local=np.asarray(vecinos_min_local, dtype=np.int64),
                meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
            )
            if candidatos_all is not None:
                arrays["candidatos_all"] = np.asarray(candidatos_all, dtype=np.int64)
            if candidatos_min is not None:
                arrays["candidatos_min"] = np.asarray(candidatos_min, dtype=np.int64)

            np.savez_compressed(path, **arrays)
            return path
        except Exception:
            return None

    # utilidades
    def clear_key(self, X: np.ndarray, dataset: str, k: int, metric: str,
                  extra: Optional[Dict[str, Any]] = None) -> bool:
        key = self.make_key(X, dataset, k, metric, extra)
        path = self.path_for(key)
        if path.exists():
            path.unlink(missing_ok=True)
            return True
        return False

    def clear_all(self) -> int:
        cnt = 0
        for p in self.cache_dir.glob("*.npz"):
            try:
                p.unlink(missing_ok=True)
                cnt += 1
            except Exception:
                pass
        return cnt
