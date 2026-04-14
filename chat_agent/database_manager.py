import sys
from langchain_community.utilities import SQLDatabase
from air_quality.exception.exception import AirQualityException
from .config import Config

class DatabaseManager:
    def __init__(self):
        try:
            self.uri = Config.get_db_uri()
            self.db = None
        except Exception as e:
            raise AirQualityException(e, sys)

    def connect(self) -> SQLDatabase:
        try:
            if not self.db:
                self.db = SQLDatabase.from_uri(self.uri, engine_args={"connect_args": {"connect_timeout": 5}})
            return self.db
        except Exception as e:
            print(f"Conection Error to DB: {e}")
            return None