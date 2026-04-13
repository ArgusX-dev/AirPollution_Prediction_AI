import os,sys
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
from air_quality.exception.exception import AirQualityException
from air_quality.logging.logger import logger


load_dotenv()

def get_db_engine():
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT",'5432')
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME",'postgres')

    try:
        logger.info(f"Connecting to PostgreSQL database {db_name}")
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        engine = create_engine(db_url)
        logger.info('Connected to PostgreSQL')
        return engine
    except Exception as e:
        raise AirQualityException(str(e),sys)

