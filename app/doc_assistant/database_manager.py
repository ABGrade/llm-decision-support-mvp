import os
import json
import logging
from tqdm import tqdm
from qdrant_client import QdrantClient, models

from . import config
from . import embedding_service

_client = None
logger = logging.getLogger(__name__)

def connect_to_db():
    global _client
    if _client is None:
        logger.info("Подключаемся к базе данных")
        _client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        logger.info("Подключено успешно")
    else:
        logger.info("Подключение уже было установлено")
    return _client

def search_in_db(force=False):
    client = connect_to_db()
    try:
        client.get_collection(collection_name=config.COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Поиск коллекции не удался {e}")
        return None
    query_vector = embedding_service.get_request_vector(force)
    embedding_service.cleanup_embedding_model()
    logger.info("Поиск в бд...")
    search_results = client.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=query_vector,
        limit=config.TOP_K,
        with_payload=True
    )
    logger.info("Поиск завершен.")
    return search_results

def upload_to_db():
    client = connect_to_db()
    try:
        client.get_collection(collection_name=config.COLLECTION_NAME)
    except Exception as e:
        logger.warning("Поиск коллекции не удался, будет создана новая")
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=config.EMB_DIMENSIONS,
                distance=models.Distance.COSINE
            ),
            timeout=15
        )
        logger.info(f"Коллекция '{config.COLLECTION_NAME}' с размером вектора {config.EMB_DIMENSIONS} успешно создана.")
    logger.info(f"Начинаем загрузку данных из файла '{config.DATABASE}' в Qdrant...")
    with open(config.DATABASE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f)

    points_to_upload = []
    with open(config.DATABASE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Загрузка в Qdrant")):
            if not line.strip():
                continue
            data = json.loads(line)
            point = models.PointStruct(
                id=i + 1,
                vector=data["vector"],
                payload={
                    "text": data["text"],
                    "metadata": data["metadata"]
                }
            )
            points_to_upload.append(point)
            if len(points_to_upload) >= 100:
                logger.debug(f"Отсылаем в БД, обработано {i} объектов")
                client.upsert(
                    collection_name=config.COLLECTION_NAME,
                    points=points_to_upload,
                    wait=False
                )
                points_to_upload = []
    if points_to_upload:
        logger.debug(f"Отсылаем оставшиеся в БД")
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points_to_upload,
            wait=True
        )
    logger.info("\nЗагрузка данных в Qdrant успешно завершена!")

