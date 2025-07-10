import logging
import argparse

from doc_assistant import logging_config
from doc_assistant import forming_answer

logging_config.setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("--- Начинаем формирование ответа ---")
    forming_answer.proceed()