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
                custom_schema = """
                    CREATE TABLE weather_pollution 
                    ( 
                        id                 INTEGER, 
                        date_hour          TIMESTAMP, 
                        temperature_c      FLOAT, 
                        humidity_pct       FLOAT, 
                        pressure_hpa       FLOAT, 
                        wind_speed_ms      FLOAT, 
                        wind_direction_deg FLOAT, 
                        cloudiness_pct     FLOAT, 
                        aqi_general        FLOAT, 
                        co                 FLOAT, 
                        no2                FLOAT, 
                        o3                 FLOAT, 
                        pm2_5              FLOAT, 
                        pm10               FLOAT, 
                        risk_category      VARCHAR, 
                        risk_severity      INTEGER, 
                        main_pollutant     VARCHAR, 
                        register_date      TIMESTAMP
                    );
                                """

                self.db = SQLDatabase.from_uri(
                    self.uri,
                    include_tables=["weather_pollution"],
                    custom_table_info={"weather_pollution": custom_schema},
                    engine_args={"connect_args": {"connect_timeout": 5}}
                )
            return self.db
        except Exception as e:
            print(f"Conection Error to DB: {e}")
            return None