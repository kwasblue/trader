import sqlite3
import pandas as pd
from loggers.logger import Logger
from utils.configloader import ConfigLoader
from threading import Lock

class DataStore:
    def __init__(self, db: str) -> None:
        self.db = db
        self.conn = None
        self.cursor = None
        self.config = ConfigLoader().load_config()
        self.write_lock = Lock()
        self.log_dir = self.config["folders"]["logs"]
        self.logger = Logger('app.log', 'datastore', log_dir=self.log_dir).get_logger()

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

    def sanitize_column_name(self, name):
        return name.replace(" ", "_").replace("-", "_").lower()

    def create_database(self, table_name: str, dataframe: pd.DataFrame):
        try:
            type_mapping = {
                'object': 'TEXT',
                'int64': 'INTEGER',
                'float64': 'REAL',
                'datetime64[ns]': 'TEXT',
                'bool': 'BOOLEAN'
            }

            columns_definition = [
                'id INTEGER PRIMARY KEY',
                'symbol TEXT',
                'date TEXT',
            ]

            for col in dataframe.columns:
                sanitized_col = self.sanitize_column_name(col)
                if sanitized_col not in ['symbol', 'date']:
                    dtype = dataframe[col].dtype
                    sql_type = type_mapping.get(str(dtype), 'TEXT')
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

    def fill_database(self, table_name: str, data: pd.DataFrame):
        with self.write_lock:
            self.create_database(table_name, data)
            try:
                data.columns = data.columns.str.replace(' ', '_').str.lower()
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

    def get_data_base(self, table_name: str) -> pd.DataFrame:
        try:
            if self.cursor is None:
                raise ValueError("Cursor is not initialized.")

            self.cursor.execute(f'SELECT * FROM {table_name}')
            rows = self.cursor.fetchall()
            columns = [description[0] for description in self.cursor.description]
            self.logger.info(f"Retrieved {len(rows)} rows from '{table_name}'.")
            return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            self.logger.error(f"Error retrieving data from '{table_name}': {e}")
            return pd.DataFrame()

    def get_data_by_symbol(self, table_name: str, symbol: str) -> pd.DataFrame:
        try:
            if self.cursor is None:
                raise ValueError("Cursor is not initialized.")

            self.cursor.execute(f"SELECT * FROM {table_name} WHERE symbol = ?", (symbol,))
            rows = self.cursor.fetchall()
            columns = [description[0] for description in self.cursor.description]
            self.logger.info(f"Retrieved {len(rows)} rows for symbol '{symbol}' from '{table_name}'.")
            return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            self.logger.error(f"Error retrieving data for symbol '{symbol}': {e}")
            return pd.DataFrame()
