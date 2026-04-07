import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger is instantiated multiple times
    if not logger.handlers:
        file_handler = RotatingFileHandler(
            LOGS_DIR / log_file, maxBytes=5*1024*1024, backupCount=3
        )
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

prediction_logger = setup_logger("predictions", "predictions.log", level=logging.INFO)
error_logger = setup_logger("errors", "errors.log", level=logging.ERROR)
system_logger = setup_logger("system", "system.log", level=logging.INFO)
