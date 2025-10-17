# Utils.py
import numpy as np
import pandas as pd
import json
import os
import hashlib
import datetime
from typing import Optional, Dict, Any


class Utils:
    # --------------------- Utilidades ---------------------
    def reset_logs(self):
        """Resetea logs por CLASE, por MUESTRA y metadatos."""
        self.logs_por_clase = []
        self.logs_por_muestra = []
        self.meta_experimento = {}
        self._meta = {}
        # diagnósticos (solo los que quedan vigentes con LSD)
        self._diag_densidad = None

    def exportar_log_csv(self, path_salida):
        """Exporta el log por CLASE."""
        if not self.logs_por_clase:
            print("⚠️ No hay log POR CLASE para exportar.")
            return
        pd.DataFrame(self.logs_por_clase).to_csv(path_salida, index=False)
        print(f"📁 Log por clase guardado en: {path_salida}")

    def exportar_log_muestras_excel(self, path_salida):
        """Exporta el log POR MUESTRA a un archivo Excel (.xlsx)."""
        if not self.logs_por_muestra:
            print("⚠️ No hay log POR MUESTRA para exportar.")
            return

        df = pd.DataFrame(self.logs_por_muestra)

        # Serializar listas o arrays a JSON para evitar errores en Excel
        cols_json = (
            "vecinos_all", "clase_vecinos_all", "dist_all",
            "vecinos_min", "dist_vecinos_min"
        )

        for col in cols_json:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, tuple, np.ndarray))
                    else v
                )

        # Guardar en formato Excel
        with pd.ExcelWriter(path_salida, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Log_Muestras", index=False)

        print(f"📘 Log por muestra guardado en: {path_salida}")

    def exportar_log_muestras_csv(self, path_salida):
        """Exporta el log POR MUESTRA."""
        if not self.logs_por_muestra:
            print("⚠️ No hay log POR MUESTRA para exportar.")
            return
        df = pd.DataFrame(self.logs_por_muestra)

        # Serializar listas en JSON para columnas complejas
        cols_json = (
            "vecinos_all", "clase_vecinos_all", "dist_all",
            "vecinos_min", "dist_vecinos_min"
        )

        for col in cols_json:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, tuple, np.ndarray))
                    else v
                )
        df.to_csv(path_salida, index=False)
        print(f"📁 Log por muestra guardado en: {path_salida}")

    # --------------------- Cálculos auxiliares ---------------------
    def _dist(self, A, b):
        """Distancia euclídea (usa las primeras 3 dims si modo_espacial='3d')."""
        if getattr(self, "modo_espacial", "2d") == '3d':
            return np.linalg.norm(A[:, :3] - b[:3], axis=1)  # solo primeras 3 dimensiones
        return np.linalg.norm(A - b, axis=1)  # todas las dimensiones

    # --------------------- Logger por muestra ---------------------
    @staticmethod
    def _to_cls_scalar(v):
        """Convierte etiqueta a tipo serializable estable (int si es entero; si no, str)."""
        try:
            arr = np.array(v)
            if np.issubdtype(arr.dtype, np.integer):
                return int(arr.item() if arr.shape == () else v)
        except Exception:
            pass
        try:
            return v.item()
        except Exception:
            return str(v)

    # --- helper local para claves dinámicas seguras ---
    def _kv_with_pct(self, base: str, pct, val):
        """
        Devuelve (clave, valor) donde la clave incluye el percentil.
        - pct=None => sufijo 'none'
        - val=None => se guarda None (no castear a float)
        """
        sufijo = "none" if pct is None else str(int(pct))
        key = f"{base}_{sufijo}"
        value = None if val is None else float(val)
        return key, value

    def _log_muestra(
        self,
        i,                      # índice en X_min
        X, X_min,               # matrices originales y minoritaria
        y,                      # etiquetas globales (para clases de vecinos)
        idxs_min_global,        # mapeo X_min[i] -> índice global en X
        comb,                   # máscara de filtrado por muestra
        riesgo, densidades,     # arrays
        entropias, proporciones_min,  # arrays o None
        pureza_mask, densidad_mask,   # máscaras booleanas
        umb_ent, umb_den,       # umbrales (float o None)
        vecinos_all_global,     # [n_min, k] índices globales en X
        vecinos_min_global,     # [n_min, k] índices globales minoritarios
        vecinos_validos_counts, # array de conteos válidos por percentil_dist
        dist_thr_por_muestra,   # array thresholds por muestra
        gen_from_counts,        # dict: idx_global -> sintéticas desde esa semilla
        last_delta_by_seed,     # dict: idx_global -> último delta
        last_neighbor_by_seed   # dict: idx_global -> último vecino z (idx global)
    ):
        seed_idx_global = int(idxs_min_global[i])

        # Vecinos (globales)
        v_all = list(map(int, vecinos_all_global[i].tolist()))
        v_min = list(map(int, vecinos_min_global[i].tolist()))
        # Clases de vecinos_all
        cls_all = [self._to_cls_scalar(y[idx]) for idx in v_all]

        # Distancias (opcionales, euclídeas para depuración)
        if getattr(self, "guardar_distancias", False):
            xi = X_min[i]
            d_all = self._dist(X[v_all], xi).tolist() if len(v_all) else []
            d_min = self._dist(X[v_min], xi).tolist() if len(v_min) else []
            d_vecinos_min = d_min[:]  # alias explícito
        else:
            d_all = None
            d_min = None
            d_vecinos_min = None

        # Claves dinámicas con percentiles configurados
        k_dist, v_dist = self._kv_with_pct(
            "percentil_dist",
            getattr(self, "percentil_dist", None),
            getattr(self, "getUmbralDistancia", lambda: None)()
        )
        k_den,  v_den  = self._kv_with_pct("percentil_densidad", getattr(self, "percentil_densidad", None), umb_den)
        k_ent,  v_ent  = self._kv_with_pct("percentil_entropia", getattr(self, "percentil_entropia", None), umb_ent)

        # Registro por muestra
        rec = {
            "dataset": getattr(self, "nombre_dataset", "unknown"),
            "idx_global": seed_idx_global,
            "clase_objetivo": None,
            "is_filtrada": bool(comb[i]),
            "k": getattr(self, "k", None),

            # Percentiles usados
            k_dist: v_dist,
            k_den:  v_den,
            k_ent:  v_ent,

            "criterio_pureza": getattr(self, "criterio_pureza", None),
            "riesgo": float(riesgo[i]),
            "densidad": float(densidades[i]),
            "entropia": None if entropias is None else float(entropias[i]),
            "proporcion_min": None if proporciones_min is None else float(proporciones_min[i]),
            "pasa_pureza": bool(pureza_mask[i]),
            "pasa_densidad": bool(densidad_mask[i]),

            # Vecinos y distancias (para auditoría)
            "vecinos_all": v_all,
            "clase_vecinos_all": cls_all,
            "dist_all": d_all,                 # euclídeas (debug)
            "vecinos_min": v_min,
            "dist_vecinos_min": d_vecinos_min, # euclídeas (debug)

            # Diagnóstico del threshold local por percentil
            "vecinos_validos_por_percentil": int(vecinos_validos_counts[i]),
            "thr_dist_percentil": float(dist_thr_por_muestra[i]),

            # Uso en síntesis
            "synthetics_from_this_seed": int(gen_from_counts.get(seed_idx_global, 0)),
            "last_delta": last_delta_by_seed.get(seed_idx_global, None),
            "last_neighbor_z": last_neighbor_by_seed.get(seed_idx_global, None),

            "timestamp": pd.Timestamp.now().isoformat(),
        }

        self.logs_por_muestra.append(rec)

    def _registrar_logs_sin_sinteticas(
        self,
        X, y, X_min, idxs_min_global,
        comb, riesgo, densidades, entropias, proporciones_min,
        pureza_mask, densidad_mask,
        umb_ent, umb_den,
        vecinos_all_global, vecinos_min_global,
        vecinos_validos_counts, dist_thr_por_muestra
    ):
        """
        Registra el log POR MUESTRA cuando no se generan sintéticos (por tope, falta de candidatas,
        falta de vecinos válidos, etc.). No modifica X ni y.
        """
        # Asegura consistencia de longitudes
        n_min = len(X_min)
        assert all(len(arr) == n_min for arr in [
            comb, riesgo, densidades, pureza_mask, densidad_mask,
            vecinos_all_global, vecinos_min_global,
            vecinos_validos_counts, dist_thr_por_muestra
        ]), "Dimensiones inconsistentes al registrar logs."

        # Recorre todas las semillas minoritarias y vuelca un registro 'en blanco' de síntesis
        for i in range(n_min):
            seed_idx_global = int(idxs_min_global[i])

            # Clases de vecinos_all (para auditoría)
            v_all = list(map(int, vecinos_all_global[i].tolist()))
            v_min = list(map(int, vecinos_min_global[i].tolist()))
            cls_all = [self._to_cls_scalar(y[idx]) for idx in v_all]

            # Distancias euclídeas (debug); se respeta el flag guardar_distancias
            if getattr(self, "guardar_distancias", False):
                xi = X_min[i]
                d_all = self._dist(X[v_all], xi).tolist() if len(v_all) else []
                d_min = self._dist(X[v_min], xi).tolist() if len(v_min) else []
            else:
                d_all = None
                d_min = None

            # Claves dinámicas de percentiles
            k_dist, v_dist = self._kv_with_pct(
                "percentil_dist",
                getattr(self, "percentil_dist", None),
                getattr(self, "getUmbralDistancia", lambda: None)()
            )
            k_den,  v_den  = self._kv_with_pct("percentil_densidad", getattr(self, "percentil_densidad", None), umb_den)
            k_ent,  v_ent  = self._kv_with_pct("percentil_entropia", getattr(self, "percentil_entropia", None), umb_ent)

            rec = {
                "dataset": getattr(self, "nombre_dataset", "unknown"),
                "idx_global": seed_idx_global,
                "clase_objetivo": None,             # se completa en multiclase
                "is_filtrada": bool(comb[i]),
                "k": getattr(self, "k", None),

                # Percentiles usados
                k_dist: v_dist,
                k_den:  v_den,
                k_ent:  v_ent,

                "criterio_pureza": getattr(self, "criterio_pureza", None),
                "riesgo": float(riesgo[i]),
                "densidad": float(densidades[i]),
                "entropia": None if entropias is None else float(entropias[i]),
                "proporcion_min": None if proporciones_min is None else float(proporciones_min[i]),
                "pasa_pureza": bool(pureza_mask[i]),
                "pasa_densidad": bool(densidad_mask[i]),

                # Vecinos y distancias (auditoría)
                "vecinos_all": v_all,
                "clase_vecinos_all": cls_all,
                "dist_all": d_all,
                "vecinos_min": v_min,
                "dist_vecinos_min": d_min,

                # Diagnóstico de threshold local por percentil (no hubo síntesis)
                "vecinos_validos_por_percentil": int(vecinos_validos_counts[i]),
                "thr_dist_percentil": float(dist_thr_por_muestra[i]) if np.isfinite(dist_thr_por_muestra[i]) else None,

                # Uso en síntesis (cero)
                "synthetics_from_this_seed": 0,
                "last_delta": None,
                "last_neighbor_z": None,

                "timestamp": pd.Timestamp.now().isoformat(),
            }
            self.logs_por_muestra.append(rec)

    # --------------------- Helpers de caché / hashing (compartidos) ---------------------
    @staticmethod
    def sha1_bytes(b: bytes) -> str:
        h = hashlib.sha1()
        h.update(b)
        return h.hexdigest()

    @staticmethod
    def sha1_text(s: str) -> str:
        return Utils.sha1_bytes(s.encode("utf-8"))

    @staticmethod
    def hash_ndarray(arr) -> Optional[str]:
        if arr is None:
            return None
        a = np.asarray(arr)
        h = hashlib.sha1()
        h.update(str(a.shape).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(a.tobytes(order="C"))
        return h.hexdigest()

    @staticmethod
    def now_iso() -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def make_key_v2(
        X_ref: np.ndarray,
        dataset: str,
        k: int,
        metric: str,
        extra: Optional[Dict[str, Any]] = None,
        base_dir: str = ".pcsmote_cache_v2"
    ) -> str:
        """
        Genera una clave de carpeta determinística a partir de los metadatos del artefacto.
        Crea el directorio destino si no existe y devuelve la ruta final.
        """
        X_ref = np.asarray(X_ref)
        meta = {
            "dataset": dataset,
            "k": int(k),
            "metric": metric,
            "shape": tuple(X_ref.shape),
            "dtype": str(X_ref.dtype),
            "extra": extra or {}
        }
        key = Utils.sha1_text(json.dumps(meta, sort_keys=True))
        path = os.path.join(base_dir, key)
        Utils.ensure_dir(path)
        return path

    @staticmethod
    def load_npy_if_exists(key: str, fname: str) -> Optional[np.ndarray]:
        """
        Carga un .npy si existe. 'key' puede ser ruta de carpeta o hash.
        """
        path = os.path.join(key, fname) if os.path.isdir(key) else os.path.join(".pcsmote_cache_v2", key, fname)
        if os.path.exists(path):
            return np.load(path, allow_pickle=False)
        return None

    @staticmethod
    def atomic_save_npy_if_exists(
        key: str,
        fname: str,
        arr: np.ndarray,
        meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guarda un único .npy en la carpeta 'key' de forma segura. Si ya existe, no sobreescribe.
        Si 'meta' se provee, también escribe/actualiza meta.json (merge superficial).
        Devuelve la ruta final del archivo guardado o existente.
        """
        dirpath = key if os.path.isdir(key) else os.path.join(".pcsmote_cache_v2", key)
        Utils.ensure_dir(dirpath)
        fpath = os.path.join(dirpath, fname)

        if not os.path.exists(fpath):
            tmp = fpath + ".tmp"
            np.save(tmp, np.asarray(arr))
            os.replace(tmp, fpath)

        if meta:
            meta_path = os.path.join(dirpath, "meta.json")
            old = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                except Exception:
                    old = {}
            old.update(meta)
            tmp_meta = meta_path + ".tmp"
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(old, f, ensure_ascii=False, indent=2)
            os.replace(tmp_meta, meta_path)

        return fpath

    @staticmethod
    def atomic_save_npy_and_meta(key: str, files: dict[str, np.ndarray], meta: dict):
        """
        Guarda varios .npy + meta.json de forma atómica.
        IMPORTANTE: usa file objects para evitar que np.save agregue '.npy' automáticamente
        a archivos temporales (p.ej. 'foo.npy.tmp' -> 'foo.npy.tmp.npy').
        """
        # Resolver carpeta destino (key puede ser ruta absoluta o hash)
        dirpath = key if os.path.isdir(key) else os.path.join(".pcsmote_cache_v2", key)
        os.makedirs(dirpath, exist_ok=True)

        # 1) Escribir cada .npy a archivo temporal y hacer replace atómico
        for fname, arr in files.items():
            final_path = os.path.join(dirpath, fname)           # ej: densidades.npy
            tmp_path   = final_path + ".tmp"                    # ej: densidades.npy.tmp

            # limpiar temporales huérfanos previos
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            # escribir NPY en el archivo temporal SIN que numpy agregue '.npy'
            with open(tmp_path, "wb") as f:
                np.save(f, np.asarray(arr), allow_pickle=False)

            # reemplazo atómico (Windows/Linux)
            os.replace(tmp_path, final_path)

        # 2) Guardar meta.json de forma atómica
        meta_path = os.path.join(dirpath, "meta.json")
        tmp_meta  = meta_path + ".tmp"
        try:
            if os.path.exists(tmp_meta):
                os.remove(tmp_meta)
        except Exception:
            pass

        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        os.replace(tmp_meta, meta_path)


    # ----------- Extras utilitarios no destructivos (opcionales) -----------
    @staticmethod
    def read_meta_json(key: str) -> Optional[Dict[str, Any]]:
        """Lee meta.json si existe y lo devuelve como dict."""
        dirpath = key if os.path.isdir(key) else os.path.join(".pcsmote_cache_v2", key)
        meta_path = os.path.join(dirpath, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    @staticmethod
    def update_meta_json(key: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge superficial de metadatos existentes con 'updates'.
        Devuelve el dict final escrito.
        """
        dirpath = key if os.path.isdir(key) else os.path.join(".pcsmote_cache_v2", key)
        Utils.ensure_dir(dirpath)
        meta_path = os.path.join(dirpath, "meta.json")

        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        meta.update(updates or {})
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp_meta, meta_path)
        return meta
