# app/database/db_manager.py
import sqlite3
from sqlite3 import Error
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path="data/database.db"):
        self.db_path = db_path
        self.init_db()

    def create_connection(self):
        """Crear y retornar una conexión a la base de datos."""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Error as e:
            print(f"Error al conectar a la base de datos: {e}")
            return None

    def init_db(self):
        """Inicializar la base de datos y crear las tablas necesarias."""
        conn = self.create_connection()
        if conn is not None:
            try:
                c = conn.cursor()
                # Crear tabla de usuarios
                c.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL,
                        full_name TEXT NOT NULL,
                        login_attempts INTEGER DEFAULT 0,
                        last_attempt TIMESTAMP
                    )
                ''')
                
                # Insertar usuario inicial
                c.execute('''
                    INSERT OR IGNORE INTO users (username, password, full_name)
                    VALUES (?, ?, ?)
                ''', ('uspatent', 'uspatent', 'usuario generico'))
                
                conn.commit()
            except Error as e:
                print(f"Error al inicializar la base de datos: {e}")
            finally:
                conn.close()

    def check_login_attempts(self, username: str) -> tuple:
        """Verificar intentos de login para un usuario."""
        conn = self.create_connection()
        if conn is not None:
            try:
                c = conn.cursor()
                c.execute('''
                    SELECT login_attempts, last_attempt
                    FROM users
                    WHERE username = ?
                ''', (username,))
                return c.fetchone()
            finally:
                conn.close()
        return None

    def reset_login_attempts(self, username: str):
        """Resetear contador de intentos de login."""
        conn = self.create_connection()
        if conn is not None:
            try:
                c = conn.cursor()
                c.execute('''
                    UPDATE users
                    SET login_attempts = 0
                    WHERE username = ?
                ''', (username,))
                conn.commit()
            finally:
                conn.close()

    def verify_credentials(self, username: str, password: str) -> tuple:
        """Verificar credenciales de usuario."""
        conn = self.create_connection()
        if conn is not None:
            try:
                c = conn.cursor()
                c.execute('''
                    SELECT *
                    FROM users
                    WHERE username = ? AND password = ?
                ''', (username, password))
                return c.fetchone()
            finally:
                conn.close()
        return None

    def increment_login_attempts(self, username: str) -> int:
        """Incrementar contador de intentos fallidos y retornar número actual."""
        conn = self.create_connection()
        if conn is not None:
            try:
                c = conn.cursor()
                c.execute('''
                    UPDATE users
                    SET login_attempts = login_attempts + 1,
                        last_attempt = datetime('now')
                    WHERE username = ?
                ''', (username,))
                
                c.execute('''
                    SELECT login_attempts
                    FROM users
                    WHERE username = ?
                ''', (username,))
                
                attempts = c.fetchone()
                conn.commit()
                return attempts[0] if attempts else 0
            finally:
                conn.close()
        return 0