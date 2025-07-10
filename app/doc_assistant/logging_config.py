import logging

from . import config

def setup_logging():
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    logging.getLogger("llama_cpp").setLevel(logging.WARNING)
    logging.getLogger(__name__).info("Logging configured.")