import logging
import os,sys
from datetime import datetime

LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
logs_path = os.path.join(LOGS_DIR, LOG_FILE)

logging.basicConfig(
                    format='[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(logs_path), logging.StreamHandler(sys.stdout)],
                    force=True
                    )
logger = logging.getLogger(__name__)


