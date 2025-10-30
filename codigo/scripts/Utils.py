# Utils.py
import numpy as np
import pandas as pd
import json
import os, re
import hashlib
import datetime
from typing import Optional, Dict, Any
# Para el formato condicional en Excel
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import FormulaRule


import traceback
from pathlib import Path

_ILLEGAL_NAME = re.compile(r'[<>:"\\|?*\x00]')

class Utils:
    # --------------------- Utilidades ---------------------
   

    def _loggable_random_state(self):
        if isinstance(self._seed_init, (int, np.integer)):
            return int(self._seed_init)
        if self._seed_init is None:
            return None
        return str(self._seed_init)

    def _motivo(self, code, detail=None):
        """
        Normaliza motivos de corte.
        """
        if detail is not None:
            return f"{code}:{detail}"
        return code

    def _push_trace(self, **kv):
        """
        Agrega un evento a self.debug_trace con timestamp.
        """
        kv = dict(kv)
        kv.setdefault("ts", pd.Timestamp.now().isoformat())
        self.debug_trace.append(kv)

    def reset_logs(self):
        """Resetea logs por CLASE, por MUESTRA y metadatos."""
        self.logs_por_clase = []
        self.logs_por_muestra = []
        self.meta_experimento = {}
        self._meta = {}
        # diagnósticos (solo los que quedan vigentes con LSD)
        self._diag_densidad = None

    @staticmethod
    def tag_p(criterio: str) -> str:
        return "entropia" if criterio == "entropia" else "Pproporcion"

    @staticmethod
    def safe_token(s: str) -> str:
        """Sanitiza un fragmento para NOMBRE DE ARCHIVO (no toca la ruta)."""
        s = str(s).replace('\r','').replace('\n','').replace('\t','').strip()
        s = s.rstrip(' .')                          # sin espacio/punto de cierre
        return re.sub(r'[^A-Za-z0-9._-]+', '', s)  # solo seguro en Windows

    def atomic_save_csv_strict(self, df: pd.DataFrame, path_str: str, fallback_dir="C:/datasets") -> str:
        """Crea carpeta, sanea filename, maneja long-path y guarda atómico."""
        p = Path(path_str)
        # saneo SOLO el filename
        name = Utils.safe_token(p.name)
        if re.search(r'[<>:"\\|?*\x00]', name):
            raise ValueError(f"Filename inválido tras sanitizar: {repr(name)}")
        p = p.with_name(name)

        # long-path fallback si hiciera falta
        abs_guess = p.resolve(strict=False)
        if len(str(abs_guess)) > 240:  # margen para Windows sin long paths
            base = Path(fallback_dir); base.mkdir(parents=True, exist_ok=True)
            p = base / p.name
            print(f"↪ Ruta larga: guardo en {p}")

        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, p)
        return str(p)


    def _debug_path(self, path_in) -> None:
        p = Path(path_in)
        print("=== DEBUG PATH ===")
        print("repr(path):", repr(str(path_in)))
        print("cwd:", os.getcwd())
        try:
            abs_guess = p.resolve(strict=False)
            print("abs path:", str(abs_guess))
            print("abs len:", len(str(abs_guess)))
        except Exception as e:
            print("resolve() lanzó:", type(e).__name__, e)

        print("parent:", str(p.parent))
        print("parent exists?", p.parent.exists(), "is_dir?", p.parent.is_dir())

        # ✅ Chequear SOLO el nombre de archivo (sin '/')
        name = p.name
        print("filename:", repr(name))
        ilegales = re.findall(r'[<>:"\\|?*\x00]', name)
        if ilegales:
            print("❗ illegal char(s) en filename:", ilegales)
        if name.endswith(' ') or name.endswith('.'):
            print("❗ filename termina con espacio o punto")

        # ¿parent escribible?
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            probe = p.parent / "_probe_write.tmp"
            with open(probe, "wb") as f:
                f.write(b"ok")
            os.replace(probe, p.parent / "_probe_write.ok")
            print("✅ Escritura en parent OK")
        except Exception as e:
            print("❌ NO se pudo escribir en parent:", type(e).__name__, e)
            traceback.print_exc()
        print("==================")

    def exportar_log_csv(self, path_salida):
        """Exporta el log por CLASE con diagnóstico."""
        if not self.logs_por_clase:
            print("⚠️ No hay log POR CLASE para exportar. CSV")
            return

        p = Path(path_salida)
        # self._debug_path(p)  # DEBUG
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        pd.DataFrame(self.logs_por_clase).to_csv(tmp, index=False)
        os.replace(tmp, p)
        print(f"📁 Log por clase guardado en: {p}")

    def exportar_log_muestras_excel_debug(self, path_salida,
                                        resaltar_no_filtradas=True,
                                        indices_resaltar=None):
        """
        Versión instrumentada para aislar fallos de escritura Excel.
        """
        print("==== DEBUG exportar_log_muestras_excel ====")
        print("→ path_salida repr:", repr(path_salida))
        print("→ cwd:", os.getcwd())

        try:
            # -- Comprobación de estructura
            p = Path(path_salida)
            print("→ absolute:", str(p.resolve(strict=False)))
            print("→ exists parent:", p.parent.exists(), "is_dir:", p.parent.is_dir())
            print("→ filename:", p.name, "len(abs):", len(str(p.resolve(strict=False))))
            for ch in '<>:"/\\|?*\x00':
                if ch in path_salida:
                    print("⚠️ Illegal char found:", repr(ch))

            # -- Datos
            if not self.logs_por_muestra:
                print("⚠️ No hay logs.")
                return
            df = pd.DataFrame(self.logs_por_muestra)

            print("→ DataFrame shape:", df.shape)
            print("→ Columns:", list(df.columns))

            # -- Serialización
            cols_json = ("vecinos_all", "clase_vecinos_all", "dist_all",
                        "vecinos_min", "dist_vecinos_min")
            for col in cols_json:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda v: json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (list, tuple, np.ndarray))
                        else v
                    )

            # -- Escritura segura
            print("→ Intentando abrir ExcelWriter...")
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print("❌ mkdir falló:", type(e).__name__, e)
                traceback.print_exc()

            try:
                with pd.ExcelWriter(str(p), engine="openpyxl") as writer:
                    hoja = "Log_Muestras"
                    df.to_excel(writer, sheet_name=hoja, index=False)
                print("✅ Escritura exitosa en:", p)
            except Exception as e:
                print("❌ Error al escribir Excel:", type(e).__name__, e)
                traceback.print_exc()
        except Exception as e_outer:
            print("❌ EXCEPCIÓN GENERAL:", type(e_outer).__name__, e_outer)
            traceback.print_exc()
        finally:
            print("==== FIN DEBUG ====")


    def exportar_log_muestras_excel(self, path_salida,
                                    resaltar_no_filtradas: bool = True,
                                    indices_resaltar=None):
        """
        Exporta el log POR MUESTRA a un Excel y aplica formato condicional:
        - Filas con is_filtrada == False → fondo rojo suave.
        - Opcional: también resalta por lista de idx_global (indices_resaltar).
        """
        if not getattr(self, "logs_por_muestra", None):
            print("⚠️ No hay log POR MUESTRA para exportar.")
            return

        df = pd.DataFrame(self.logs_por_muestra)

        # Serializar listas/arrays a JSON para que Excel no falle
        cols_json = ("vecinos_all", "clase_vecinos_all", "dist_all",
                    "vecinos_min", "dist_vecinos_min")
        for col in cols_json:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, tuple, np.ndarray))
                    else v
                )

        # Si el usuario pasó índices a resaltar, calculo máscara
        marcar_por_idx = None
        if indices_resaltar is not None and "idx_global" in df.columns:
            s = set(indices_resaltar)
            marcar_por_idx = df["idx_global"].isin(s)

        p = Path(path_salida)
        p.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(str(p), engine="openpyxl") as writer:
            hoja = "Log_Muestras"
            df.to_excel(writer, sheet_name=hoja, index=False)
            ws = writer.sheets[hoja]

            # Rango de datos (sin encabezado)
            n_rows = len(df)
            n_cols = len(df.columns)
            if n_rows == 0 or n_cols == 0:
                return

            first_row = 2
            last_row = first_row + n_rows - 1
            last_col_letter = get_column_letter(n_cols)

            # Regla 1: is_filtrada == False → rojo suave
            if resaltar_no_filtradas and "is_filtrada" in df.columns:
                col_idx = df.columns.get_loc("is_filtrada") + 1  # 1-based
                col_letter = get_column_letter(col_idx)
                red_fill = PatternFill(start_color="FFF2D7D9", end_color="FFF2D7D9", fill_type="solid")
                formula = f"=${col_letter}{first_row}=FALSE"
                ws.conditional_formatting.add(
                    f"A{first_row}:{last_col_letter}{last_row}",
                    FormulaRule(formula=[formula], fill=red_fill)
                )

            # Regla 2 (opcional): resaltar filas por idx_global
            if marcar_por_idx is not None and marcar_por_idx.any():
                aux_col_idx = n_cols + 1
                aux_col_letter = get_column_letter(aux_col_idx)
                ws.cell(row=1, column=aux_col_idx, value="__mark__")
                for r, v in enumerate(marcar_por_idx.values, start=2):
                    ws.cell(row=r, column=aux_col_idx, value=bool(v))

                yellow_fill = PatternFill(start_color="FFFFFFCC", end_color="FFFFFFCC", fill_type="solid")
                formula2 = f"=${aux_col_letter}{first_row}=TRUE"
                ws.conditional_formatting.add(
                    f"A{first_row}:{last_col_letter}{last_row}",
                    FormulaRule(formula=[formula2], fill=yellow_fill)
                )

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


