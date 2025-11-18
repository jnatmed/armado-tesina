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
        return "P" + str(criterio).lower().strip()

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

        Modificación:
        - Agrega una columna 'configuracion' (si self.nombre_configuracion existe).
        - Si la clase tiene acumulador por dataset (PCSMOTE), se acumulan
          todas las configuraciones del mismo dataset y se exportan unificadas.
        """
        if not getattr(self, "logs_por_muestra", None):
            print("⚠️ No hay log POR MUESTRA para exportar.")
            return

        # DataFrame base con los logs de ESTA ejecución
        df = pd.DataFrame(self.logs_por_muestra)

        # ───────────── Columna extra: 'configuracion' ─────────────
        # Solo una columna nueva, usando self.nombre_configuracion si existe.
        nombre_configuracion = getattr(self, "nombre_configuracion", None)
        if nombre_configuracion is not None:
            if "configuracion" not in df.columns:
                # Insertar después de 'dataset' si existe, si no al principio
                if "dataset" in df.columns:
                    idx_dataset = list(df.columns).index("dataset")
                    pos_insert = idx_dataset + 1
                else:
                    pos_insert = 0
                df.insert(pos_insert, "configuracion", str(nombre_configuracion))

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

        # ───────────── Acumulación por dataset (solo para PCSMOTE) ─────────────
        df_salida = df  # por defecto, sin acumulación

        nombre_dataset = getattr(self, "nombre_dataset", None)
        cls = type(self)

        # Si la clase tiene métodos de acumulación, asumimos que es PCSMOTE
        if hasattr(cls, "acumular_logs_por_muestra_por_dataset") and \
           hasattr(cls, "obtener_logs_por_muestra_acumulados") and \
           nombre_dataset is not None:

            # 1) acumular este bloque
            cls.acumular_logs_por_muestra_por_dataset(nombre_dataset, df)

            # 2) recuperar todos los logs acumulados para este dataset
            df_salida = cls.obtener_logs_por_muestra_acumulados(nombre_dataset)

        # Si el usuario pasó índices a resaltar, calculo máscara sobre df_salida
        marcar_por_idx = None
        if indices_resaltar is not None and "idx_global" in df_salida.columns:
            s = set(indices_resaltar)
            marcar_por_idx = df_salida["idx_global"].isin(s)

        p = Path(path_salida)
        p.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(str(p), engine="openpyxl") as writer:
            hoja = "Log_Muestras"
            df_salida.to_excel(writer, sheet_name=hoja, index=False)
            ws = writer.sheets[hoja]

            # Rango de datos (sin encabezado)
            n_rows = len(df_salida)
            n_cols = len(df_salida.columns)
            if n_rows == 0 or n_cols == 0:
                return

            first_row = 2
            last_row = first_row + n_rows - 1
            last_col_letter = get_column_letter(n_cols)

            # Regla 1: is_filtrada == False → rojo suave
            if resaltar_no_filtradas and "is_filtrada" in df_salida.columns:
                col_idx = df_salida.columns.get_loc("is_filtrada") + 1  # 1-based
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
        pureza_mask, densidad_mask, mask_riesgo,
        umb_ent, umb_den,       # umbrales (float o None)
        vecinos_all_global,     # [n_min, k] índices globales en X
        vecinos_min_global,     # [n_min, k] índices globales minoritarios
        vecinos_validos_counts, # array de conteos válidos por percentil_dist
        dist_thr_por_muestra,   # array thresholds por muestra
        gen_from_counts,        # dict: idx_global -> sintéticas desde esa semilla
        last_delta_by_seed,     # dict: idx_global -> último delta
        last_neighbor_by_seed   # dict: idx_global -> último vecino z (idx global)
    ):
        """
        Registro por muestra con esquema de columnas FIJO.
        - Sin timestamp.
        - Sin columnas dinámicas percentil_xxx_NONE / _50 / _75.
        - 'percentil_*' almacena el VALOR UMBRAL del percentil (no 25/50/75).
        - Agrega columna 'configuracion' proveniente de self.nombre_configuracion.
        """

        # índice global de la semilla en X
        seed_idx_global = int(idxs_min_global[i])

        # Vecinos globales y minoritarios
        vecinos_all_lista = []
        for idx in vecinos_all_global[i].tolist():
            vecinos_all_lista.append(int(idx))

        vecinos_min_lista = []
        for idx in vecinos_min_global[i].tolist():
            vecinos_min_lista.append(int(idx))

        # Clases de vecinos_all
        clases_vecinos_all = []
        for idx_vecino in vecinos_all_lista:
            clases_vecinos_all.append(self._to_cls_scalar(y[idx_vecino]))

        # Distancias (opcionales, euclídeas para depuración)
        if getattr(self, "guardar_distancias", False):
            xi = X_min[i]

            if len(vecinos_all_lista) > 0:
                d_all = self._dist(X[vecinos_all_lista], xi).tolist()
            else:
                d_all = []

            if len(vecinos_min_lista) > 0:
                d_min = self._dist(X[vecinos_min_lista], xi).tolist()
            else:
                d_min = []

            d_vecinos_min = d_min[:]  # alias explícito
        else:
            d_all = None
            d_vecinos_min = None

        # ───────── Umbrales asociados a los percentiles (valores numéricos) ─────────
        valor_percentil_dist = None
        valor_percentil_densidad = None
        valor_percentil_entropia = None
        valor_percentil_riesgo = None

        # umbral global de distancias usado en calcular_densidad_interseccion
        if hasattr(self, "_meta") and isinstance(self._meta, dict):
            if "umbral_densidad_global" in self._meta:
                valor_percentil_dist = self._meta["umbral_densidad_global"]
            if "umbral_riesgo_min" in self._meta:
                valor_percentil_riesgo = self._meta["umbral_riesgo_min"]

        if umb_den is not None:
            valor_percentil_densidad = float(umb_den)

        if umb_ent is not None:
            valor_percentil_entropia = float(umb_ent)

        # Construcción explícita del registro
        registro = {
            # contexto
            # "dataset": getattr(self, "nombre_dataset", "unknown"),
            # "configuracion": getattr(self, "nombre_configuracion", None),

            "idx_global": seed_idx_global,
            "clase_objetivo": None,  # se pisa desde fit_resample_multiclass
            "is_filtrada": bool(comb[i]),
            "k": int(getattr(self, "k", 0)),

            # percentiles usados (VALOR DE UMBRAL, no 25/50/75)
            "percentil_dist": valor_percentil_dist,
            "percentil_densidad": valor_percentil_densidad,
            "percentil_entropia": valor_percentil_entropia,
            "percentil_riesgo": valor_percentil_riesgo,

            # umbrales asociados (algunos redundan pero son más legibles)
            "umbral_densidad": None if umb_den is None else float(umb_den),
            "umbral_entropia": None if umb_ent is None else float(umb_ent),

            # métricas locales
            "criterio_pureza": getattr(self, "criterio_pureza", None),
            "riesgo": float(riesgo[i]),
            "densidad": float(densidades[i]),
            "entropia": None if entropias is None else float(entropias[i]),
            "proporcion_min": None if proporciones_min is None else float(proporciones_min[i]),
            "pasa_pureza": bool(pureza_mask[i]),
            "pasa_densidad": bool(densidad_mask[i]),
            "pasa_riesgo": bool(mask_riesgo[i]),
            
            # vecinos y distancias
            # "vecinos_all": vecinos_all_lista,
            # "clase_vecinos_all": clases_vecinos_all,
            # "dist_all": d_all,
            # "vecinos_min": vecinos_min_lista,
            # "dist_vecinos_min": d_vecinos_min,

            # diagnóstico de threshold de distancia por muestra
            "vecinos_validos_por_percentil": int(vecinos_validos_counts[i]),
            "thr_dist_percentil": float(dist_thr_por_muestra[i]),

            # uso en síntesis
            "synthetics_from_this_seed": int(gen_from_counts.get(seed_idx_global, 0)),
            "last_delta": last_delta_by_seed.get(seed_idx_global, None),
            "last_neighbor_z": last_neighbor_by_seed.get(seed_idx_global, None),
        }

        self.logs_por_muestra.append(registro)
