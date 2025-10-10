# Utils.py
import numpy as np
import pandas as pd
import json

class Utils:
    # --------------------- Utilidades ---------------------
    def reset_logs(self):
        """Resetea logs por CLASE, por MUESTRA y metadatos."""
        self.logs_por_clase = []
        self.logs_por_muestra = []
        self.meta_experimento = {}
        self._meta = {}
        self._S_inv_by_i = None        
        # reset diagnósticos
        self._diag_maha = None
        self._diag_umbral_maha_global = None
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
        if "maha_dists" in getattr(df, "columns", []):
            cols_json = cols_json + ("maha_dists",)

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
        # opcional: si en algún momento agregás 'maha_dists' al log por muestra:
        if "maha_dists" in getattr(df, "columns", []):
            cols_json = cols_json + ("maha_dists",)

        for col in cols_json:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, tuple, np.ndarray))
                    else v
                )
        df.to_csv(path_salida, index=False)
        print(f"📁 Log por muestra guardado en: {path_salida}")

    # Exportador opcional de diagnósticos de Mahalanobis (por semilla)
    def exportar_diag_maha_csv(self, path_salida):
        if not self._diag_maha:
            print("⚠️ No hay diagnósticos de Mahalanobis para exportar.")
            return
        rows = []
        for i, d in enumerate(self._diag_maha):
            if d is None:
                continue
            row = {"idx_local": i}
            row.update(d)
            rows.append(row)
        pd.DataFrame(rows).to_csv(path_salida, index=False)
        print(f"📁 Diagnóstico Mahalanobis guardado en: {path_salida}")

    # --------------------- Cálculos auxiliares ---------------------
    def _dist(self, A, b):
        """Distancia euclídea 2D/3D según modo_espacial."""
        if self.modo_espacial == '3d':
            return np.linalg.norm(A[:, :3] - b[:3], axis=1) # solo primeras 3 dimensiones
        return np.linalg.norm(A - b, axis=1) # todas las dimensiones

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

        # Distancias (opcionales)
        if self.guardar_distancias:
            xi = X_min[i]
            d_all = self._dist(X[v_all], xi).tolist() if len(v_all) else []
            d_min = self._dist(X[v_min], xi).tolist() if len(v_min) else []
            d_vecinos_min = d_min[:]  # alias explícito pedido
        else:
            d_all = None
            d_min = None
            d_vecinos_min = None

        # Claves dinámicas seguras
        k_dist, v_dist = self._kv_with_pct("percentil_dist", self.percentil_dist, self.getUmbralDistancia())
        k_den,  v_den  = self._kv_with_pct("percentil_densidad", self.percentil_densidad, umb_den)
        k_ent,  v_ent  = self._kv_with_pct("percentil_entropia", self.percentil_entropia, umb_ent)

        # Diagnóstico compacto por semilla (Mahalanobis)
        diag = None
        if isinstance(getattr(self, "_diag_maha", None), list) and i < len(self._diag_maha):
            diag = self._diag_maha[i] or {}
        diag_compacto = {
            "maha_ok": bool(diag.get("ok")) if isinstance(diag, dict) else None,
            "maha_fallback": bool(diag.get("fallback")) if isinstance(diag, dict) else None,
            "maha_traceS": diag.get("traceS") if isinstance(diag, dict) else None,
            "maha_lam": diag.get("lam") if isinstance(diag, dict) else None,
            "maha_rank_Sreg": diag.get("rank_Sreg") if isinstance(diag, dict) else None,
            "maha_cond_Sreg": diag.get("cond_Sreg") if isinstance(diag, dict) else None,
            "maha_min": diag.get("d_maha_min") if isinstance(diag, dict) else None,
            "maha_med": diag.get("d_maha_med") if isinstance(diag, dict) else None,
            "maha_max": diag.get("d_maha_max") if isinstance(diag, dict) else None,
            "maha_mean": diag.get("d_maha_mean") if isinstance(diag, dict) else None,
            # "maha_dists": diag.get("d_maha_list") if isinstance(diag, dict) else None,  # ← opcional
        }

        # === Mantiene orden original ===
        rec = {
            "dataset": self.nombre_dataset,
            "idx_global": seed_idx_global,
            "clase_objetivo": None,
            "is_filtrada": bool(comb[i]),
            "k": self.k,

            # Sección de percentiles (orden original respetado)
            k_dist: v_dist,
            k_den:  v_den,
            k_ent:  v_ent,

            "criterio_pureza": self.criterio_pureza,
            "riesgo": float(riesgo[i]),
            "densidad": float(densidades[i]),
            "entropia": None if entropias is None else float(entropias[i]),
            "proporcion_min": None if proporciones_min is None else float(proporciones_min[i]),
            "pasa_pureza": bool(pureza_mask[i]),
            "pasa_densidad": bool(densidad_mask[i]),

            # Vecinos y distancias
            "vecinos_all": v_all,
            "clase_vecinos_all": cls_all,
            "dist_all": d_all,
            "vecinos_min": v_min,
            "dist_vecinos_min": d_vecinos_min,

            # Diagnóstico percentil local
            "vecinos_validos_por_percentil": int(vecinos_validos_counts[i]),
            "thr_dist_percentil": float(dist_thr_por_muestra[i]),

            # Uso en síntesis
            "synthetics_from_this_seed": int(gen_from_counts.get(seed_idx_global, 0)),
            "last_delta": last_delta_by_seed.get(seed_idx_global, None),
            "last_neighbor_z": last_neighbor_by_seed.get(seed_idx_global, None),
            "timestamp": pd.Timestamp.now().isoformat(),
        }     # p.ej.: "percentil_entropia_none": None

        # anexar diagnóstico compacto sin romper el orden principal (queda al final)
        rec.update(diag_compacto)

        self.logs_por_muestra.append(rec)
