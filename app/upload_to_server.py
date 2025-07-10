import logging
import argparse

from doc_assistant import logging_config
from doc_assistant import database_manager, embedding_service

logging_config.setup_logging()
logger = logging.getLogger(__name__)

def setup_parser():
    parser = argparse.ArgumentParser(
        description="Построение векторов и отправка на сервер")
    # parser.add_argument(
    #     "-i", "--input",
    #     type=str,
    #     default=config.DATABASE,
    #     help=f"Путь к входному файлу с чанками (по умолчанию: {config.CHUNKS_FILEPATH})"
    # )
    # parser.add_argument(
    #     "-o", "--output",
    #     type=str,
    #     default=config.DATABASE,
    #     help=f"Путь к выходному файлу базы данных (по умолчанию: {config.DATABASE})"
    # )
    parser.add_argument(
        "-i", "--ignore-main",
        action='store_true',
        help=f"Не обрабатывает БД и не загружает"
    )
    parser.add_argument(
        "-v", "--vectorize",
        action='store_true',
        help=f"Только векторизация БД"
    )
    parser.add_argument(
        "-u", "--upload",
        action='store_true',
        help=f"Прямая отправка в БД без обработки"
    )
    parser.add_argument(
        "-r", "--request",
        action='store_true',
        help=f"Обработка запроса пользователя"
    )
    parser.add_argument(
        "-rf", "--request-force",
        action='store_true',
        help=f"Обработка запроса пользователя c принудительной векторизацией"
    )
    return parser

if __name__ == "__main__":
    logger.info("--- Начинаем процесс загрузки в БД Qdrant ---")

    arg_parser = setup_parser()
    args = arg_parser.parse_args()

    if not args.ignore_main:
        embedding_service.create_db_vectors()
        if args.request_force:
            logging.info("Выполнение принудительной векторизации запроса")
            embedding_service.get_request_vector()
        elif args.request:
            logging.info("Выполнение векторизации запроса")
            embedding_service.get_request_vector()
        database_manager.upload_to_db()
    else:
        if args.vectorize:
            logging.info("Выполнение векторизации данных")
            embedding_service.create_db_vectors()
        elif args.request_force:
            logging.info("Выполнение принудительной векторизации запроса")
            embedding_service.get_request_vector(force_recalculate=True)
        elif args.request:
            logging.info("Выполнение векторизации запроса")
            embedding_service.get_request_vector()
        if args.upload:
            logging.info("Выполнение загрузки данных в БД")
            database_manager.upload_to_db()

    logging.info("--- Работа утилиты завершена ---")