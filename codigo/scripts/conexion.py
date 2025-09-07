import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

class DatabaseConnection:
    def __init__(self):
        # 1) Intento cargar .env que esté junto a ESTE archivo (scripts/.env)
        here = os.path.dirname(os.path.abspath(__file__))
        env_path_local = os.path.join(here, ".env")
        if os.path.exists(env_path_local):
            load_dotenv(env_path_local)

        self.host = os.getenv("DB_HOST", "localhost")
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_NAME", "tesina_oversampling")
        self.port = int(os.getenv("DB_PORT", 3306))

        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            if self.connection.is_connected():
                self.cursor = self.connection.cursor(dictionary=True)
                print(f"✅ Conectado a {self.database} en {self.host}:{self.port}")
        except Error as e:
            print("❌ Error al conectar a MySQL:", e)

    def execute_query(self, query, params=None):
        try:
            self.cursor.execute(query, params or ())
            self.connection.commit()
            return self.cursor.rowcount
        except Error as e:
            print("❌ Error al ejecutar consulta:", e)
            return None

    def fetch_all(self, query, params=None):
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except Error as e:
            print("❌ Error al obtener datos:", e)
            return None

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("🔒 Conexión cerrada")
