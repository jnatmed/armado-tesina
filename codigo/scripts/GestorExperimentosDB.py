import numpy as np
import json

class GestorExperimentosDB:
    def __init__(self, db):
        self.db = db

    def guardar_particion(self, experimento_id: int, split: str, X, y, clear_before=True, batch_size: int = 3000):
        assert split in ("train", "test")
        if clear_before:
            self.db.exec("DELETE FROM `particion_dataset` WHERE experimento_id=%s AND split=%s", (experimento_id, split))

        X = np.asarray(X)
        y = np.asarray(y)
        rows = []
        for i, (xi, yi) in enumerate(zip(X, y)):
            rows.append((experimento_id, split, i, json.dumps(list(map(float, xi))), str(yi)))
            if len(rows) >= batch_size:
                self.db.cursor.executemany(
                    "INSERT INTO `particion_dataset` (`experimento_id`,`split`,`fila`,`features`,`target`) VALUES (%s,%s,%s,%s,%s)",
                    rows
                )
                self.db.connection.commit()
                rows.clear()
        if rows:
            self.db.cursor.executemany(
                "INSERT INTO `particion_dataset` (`experimento_id`,`split`,`fila`,`features`,`target`) VALUES (%s,%s,%s,%s,%s)",
                rows
            )
            self.db.connection.commit()

    def get_or_create_dataset_id(self, nombre, n_train=None, n_test=None, n_features=None, es_grande=0):
        sql = (
            "INSERT INTO `dataset` (`nombre`,`n_train`,`n_test`,`n_features`,`es_grande`) "
            "VALUES (%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE "
            "  `dataset_id`=LAST_INSERT_ID(`dataset_id`), "
            "  `n_train`=VALUES(`n_train`), `n_test`=VALUES(`n_test`), "
            "  `n_features`=VALUES(`n_features`), `es_grande`=VALUES(`es_grande`)"
        )
        self.db.exec(sql, (nombre, n_train, n_test, n_features, es_grande))
        return int(self.db.cursor.lastrowid)

    # Agregá aquí los otros métodos: get_or_create_modelo_id, get_or_create_config_id, upsert_experimento_y_metricas, etc.
    def get_or_create_modelo_id(self, nombre: str) -> int:
        """
        Inserta o recupera el modelo por nombre.
        """
        sql = (
            "INSERT INTO `modelo` (`nombre`) VALUES (%s) "
            "ON DUPLICATE KEY UPDATE `modelo_id`=LAST_INSERT_ID(`modelo_id`)"
        )
        self.db.exec(sql, (nombre,))
        return int(self.db.cursor.lastrowid)

    def get_or_create_config_id(self, tecnica: str, densidad, riesgo, pureza, tipo: str) -> int:
        sql = (
            "INSERT INTO `config_sobremuestreo` "
            "(`tecnica`,`densidad`,`riesgo`,`pureza`,`tipo`) "
            "VALUES (%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE `config_id`=LAST_INSERT_ID(`config_id`)"
        )
        self.db.exec(sql, (tecnica, densidad, riesgo, pureza, tipo))
        return int(self.db.cursor.lastrowid)


    def upsert_experimento_y_metricas(
        self, dataset_id:int, config_id:int, modelo_id:int,
        cv_splits:int|None, n_iter:int|None, n_jobs_search:int|None,
        search_time_sec, mejor_configuracion, source_file:str, metricas:dict
    ) -> int:
        # 1) experimento (idempotente por UNIQUE textual y triggers)
        sql_exp = (
            "INSERT INTO `experimento` "
            "(`dataset_id`,`config_id`,`modelo_id`,`cv_splits`,`n_iter`,`n_jobs_search`,"
            "`search_time_sec`,`mejor_configuracion`,`source_file`) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) AS v "
            "ON DUPLICATE KEY UPDATE "
            "`experimento_id`=LAST_INSERT_ID(`experimento_id`), "
            "`mejor_configuracion`=v.`mejor_configuracion`, "
            "`search_time_sec`=v.`search_time_sec`, "
            "`cv_splits`=v.`cv_splits`, `n_iter`=v.`n_iter`, "
            "`n_jobs_search`=v.`n_jobs_search`, `source_file`=v.`source_file`"
        )
        self.db.exec(sql_exp, (
            dataset_id, config_id, modelo_id, cv_splits, n_iter, n_jobs_search,
            search_time_sec, json.dumps(mejor_configuracion, ensure_ascii=False), source_file
        ))
        experimento_id = int(self.db.cursor.lastrowid)

        # 2) metricas (tabla separada con UNIQUE por experimento_id)
        sql_met = (
            "INSERT INTO `metricas` "
            "(`experimento_id`,`cv_f1_macro`,`cv_balanced_accuracy`,`cv_mcc`,`cv_cohen_kappa`,"
            "`test_f1_macro`,`test_balanced_accuracy`,`test_mcc`,`test_cohen_kappa`) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE "
            "`cv_f1_macro`=VALUES(`cv_f1_macro`), "
            "`cv_balanced_accuracy`=VALUES(`cv_balanced_accuracy`), "
            "`cv_mcc`=VALUES(`cv_mcc`), "
            "`cv_cohen_kappa`=VALUES(`cv_cohen_kappa`), "
            "`test_f1_macro`=VALUES(`test_f1_macro`), "
            "`test_balanced_accuracy`=VALUES(`test_balanced_accuracy`), "
            "`test_mcc`=VALUES(`test_mcc`), "
            "`test_cohen_kappa`=VALUES(`test_cohen_kappa`)"
        )
        vals = (
            experimento_id,
            metricas.get("cv_f1_macro"), metricas.get("cv_balanced_accuracy"),
            metricas.get("cv_mcc"), metricas.get("cv_cohen_kappa"),
            metricas.get("test_f1_macro"), metricas.get("test_balanced_accuracy"),
            metricas.get("test_mcc"), metricas.get("test_cohen_kappa")
        )
        self.db.exec(sql_met, vals)
        return experimento_id

