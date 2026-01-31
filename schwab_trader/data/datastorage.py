import sqlite3
import logging
import pandas as pd
from contextlib import contextmanager
from threading import RLock
from typing import Optional

# Standard library logger (no external dependencies)
logger = logging.getLogger(__name__)


class DataStore:
    """
    SQLite-based data store for market data.

    Features:
    - Thread-safe with write locking
    - Context manager support for automatic connection handling
    - Works independently (no ConfigLoader dependency required)

    Usage:
        # Option 1: Context manager (recommended)
        with DataStore('stock_base.db') as store:
            store.fill_database('AAPL_data', df)
            data = store.get_data_by_symbol('AAPL_data', 'AAPL')

        # Option 2: Manual management (backward compatible)
        store = DataStore('stock_base.db')
        store.open_db()
        store.fill_database('AAPL_data', df)
        store.close_db()
    """

    def __init__(self, db: str, use_config: bool = True) -> None:
        self.db = db
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self.write_lock = RLock()  # Reentrant lock to avoid deadlocks
        self._owns_connection = False

        # Try to use ConfigLoader for backward compatibility, but don't require it
        self.logger = logger
        if use_config:
            try:
                from utils.configloader import ConfigLoader
                from loggers.logger import Logger
                self.config = ConfigLoader().load_config()
                log_dir = self.config["folders"]["logs"]
                self.logger = Logger('app.log', 'datastore', log_dir=log_dir).get_logger()
            except Exception:
                # Fall back to standard logging if ConfigLoader fails
                pass

    def __enter__(self):
        """Context manager entry - opens database connection."""
        self.open_db()
        self._owns_connection = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes database connection."""
        self.close_db()
        return False  # Don't suppress exceptions

    def open_db(self):
        try:
            if not self.conn:
                self.logger.info("Opening database connection...")
                self.conn = sqlite3.connect(self.db, check_same_thread=False)
                self.conn.execute('PRAGMA journal_mode=WAL;')
                self.cursor = self.conn.cursor()
                self.logger.info(f"Connection initialized: {self.conn}")
            else:
                self.logger.warning("Database connection is already open.")
        except Exception as e:
            self.logger.error(f"Error opening database connection: {e}")
            raise

    def close_db(self):
        try:
            if self.cursor:
                self.cursor.close()
                self.logger.info("Cursor closed.")
            if self.conn:
                self.conn.close()
                self.logger.info("Database connection closed.")
            self.conn = None
            self.cursor = None
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")
            raise

    def commit(self):
        try:
            if self.conn:
                self.conn.commit()
                self.logger.info("Transaction committed.")
        except Exception as e:
            self.logger.error(f"Error during commit: {e}")
            self.conn.rollback()

    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize table name to prevent SQL injection."""
        import re
        # Only allow alphanumeric and underscores
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

    def sanitize_column_name(self, name):
        return name.replace(" ", "_").replace("-", "_").lower()

    def _ensure_connected(self):
        """Ensure database connection is open."""
        if self.conn is None or self.cursor is None:
            raise RuntimeError(
                "Database not connected. Use 'with DataStore(db) as store:' "
                "or call store.open_db() first."
            )

    def create_database(self, table_name: str, dataframe: pd.DataFrame):
        """Create table from dataframe schema."""
        self._ensure_connected()
        table_name = self._sanitize_table_name(table_name)

        try:
            type_mapping = {
                'object': 'TEXT',
                'int64': 'INTEGER',
                'int32': 'INTEGER',
                'float64': 'REAL',
                'float32': 'REAL',
                'datetime64[ns]': 'TEXT',
                'datetime64[ns, UTC]': 'TEXT',
                'bool': 'INTEGER',
                'boolean': 'INTEGER',
            }

            columns_definition = ['id INTEGER PRIMARY KEY']

            for col in dataframe.columns:
                sanitized_col = self.sanitize_column_name(col)
                dtype = str(dataframe[col].dtype)
                sql_type = type_mapping.get(dtype, 'TEXT')
                columns_definition.append(f"{sanitized_col} {sql_type}")

            create_table_query = f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(columns_definition)}
                )
            '''

            with self.write_lock:
                self.cursor.execute(create_table_query)
                self.commit()

            self.logger.info(f"Table '{table_name}' created or already exists.")
        except Exception as e:
            self.logger.error(f"Error creating table '{table_name}': {e}")
            raise

    def fill_database(self, table_name: str, data: pd.DataFrame):
        """Insert dataframe data into table. Creates table if it doesn't exist."""
        self._ensure_connected()
        table_name = self._sanitize_table_name(table_name)

        # Make a copy to avoid mutating original dataframe
        data = data.copy()

        with self.write_lock:
            self.create_database(table_name, data)
            try:
                # Sanitize column names
                data.columns = [self.sanitize_column_name(c) for c in data.columns]
                values_to_insert = data.to_records(index=False).tolist()
                columns = ', '.join(data.columns)
                placeholders = ', '.join(['?'] * len(data.columns))
                query = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'

                if values_to_insert:
                    self.cursor.executemany(query, values_to_insert)
                    self.commit()
                    self.logger.info(f"Inserted {len(values_to_insert)} rows into '{table_name}'.")
                else:
                    self.logger.warning(f"No valid data to insert into '{table_name}'.")
            except Exception as e:
                self.logger.error(f"Error filling table '{table_name}': {e}")
                raise

    def upsert_data(self, table_name: str, data: pd.DataFrame, key_columns: list):
        """
        Insert or update data based on key columns.

        Args:
            table_name: Target table name
            data: DataFrame to upsert
            key_columns: List of column names that form the unique key
        """
        self._ensure_connected()
        table_name = self._sanitize_table_name(table_name)

        # Make a copy to avoid mutating original dataframe
        data = data.copy()
        data.columns = [self.sanitize_column_name(c) for c in data.columns]
        key_columns = [self.sanitize_column_name(c) for c in key_columns]

        with self.write_lock:
            self.create_database(table_name, data)
            try:
                columns = list(data.columns)
                placeholders = ', '.join(['?'] * len(columns))
                update_cols = [c for c in columns if c not in key_columns]
                update_clause = ', '.join([f'{c} = excluded.{c}' for c in update_cols])

                # Create unique index if not exists
                idx_name = f"idx_{table_name}_{'_'.join(key_columns)}"
                self.cursor.execute(f'''
                    CREATE UNIQUE INDEX IF NOT EXISTS {idx_name}
                    ON {table_name} ({', '.join(key_columns)})
                ''')

                query = f'''
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT({', '.join(key_columns)}) DO UPDATE SET {update_clause}
                '''

                values_to_insert = data.to_records(index=False).tolist()
                if values_to_insert:
                    self.cursor.executemany(query, values_to_insert)
                    self.commit()
                    self.logger.info(f"Upserted {len(values_to_insert)} rows into '{table_name}'.")
            except Exception as e:
                self.logger.error(f"Error upserting to '{table_name}': {e}")
                raise

    def get_data_base(self, table_name: str) -> pd.DataFrame:
        """Get all data from a table."""
        self._ensure_connected()
        table_name = self._sanitize_table_name(table_name)

        try:
            self.cursor.execute(f'SELECT * FROM {table_name}')
            rows = self.cursor.fetchall()
            columns = [description[0] for description in self.cursor.description]
            self.logger.info(f"Retrieved {len(rows)} rows from '{table_name}'.")
            return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            self.logger.error(f"Error retrieving data from '{table_name}': {e}")
            return pd.DataFrame()

    def get_data_by_symbol(self, table_name: str, symbol: str) -> pd.DataFrame:
        """Get data for a specific symbol."""
        self._ensure_connected()
        table_name = self._sanitize_table_name(table_name)

        try:
            self.cursor.execute(f"SELECT * FROM {table_name} WHERE symbol = ?", (symbol,))
            rows = self.cursor.fetchall()
            columns = [description[0] for description in self.cursor.description]
            self.logger.info(f"Retrieved {len(rows)} rows for symbol '{symbol}' from '{table_name}'.")
            return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            self.logger.error(f"Error retrieving data for symbol '{symbol}': {e}")
            return pd.DataFrame()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        self._ensure_connected()
        table_name = self._sanitize_table_name(table_name)

        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return self.cursor.fetchone() is not None

    def list_tables(self) -> list:
        """List all tables in the database."""
        self._ensure_connected()

        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in self.cursor.fetchall()]

    def get_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        self._ensure_connected()
        table_name = self._sanitize_table_name(table_name)

        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return self.cursor.fetchone()[0]

    def delete_data(self, table_name: str, where_clause: str = None, params: tuple = None):
        """
        Delete data from a table.

        Args:
            table_name: Target table
            where_clause: Optional WHERE clause (without 'WHERE' keyword)
            params: Parameters for the WHERE clause
        """
        self._ensure_connected()
        table_name = self._sanitize_table_name(table_name)

        with self.write_lock:
            try:
                if where_clause:
                    query = f"DELETE FROM {table_name} WHERE {where_clause}"
                    self.cursor.execute(query, params or ())
                else:
                    self.cursor.execute(f"DELETE FROM {table_name}")
                self.commit()
                self.logger.info(f"Deleted rows from '{table_name}'.")
            except Exception as e:
                self.logger.error(f"Error deleting from '{table_name}': {e}")
                raise
