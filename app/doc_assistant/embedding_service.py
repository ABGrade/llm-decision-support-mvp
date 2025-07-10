import os
import json
import logging
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from . import config

_model = None
logger = logging.getLogger(__name__)

def _get_embedding_model():
    """
    Загружает модель эмбеддингов в память (если еще не загружена)
    """
    global _model
    if _model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Загрузка модели эмбеддингов '{config.EMB_MODEL_NAME}' на '{device}'...")
        try:
            _model = SentenceTransformer(
                config.EMB_MODEL_NAME,
                device=device,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load model from '{config.EMB_MODEL_NAME}' : {e}")
            raise
        logging.info("Модель эмбеддингов успешно загружена.")
    return _model

def get_request_vector(force_recalculate=False):
    """
    "Умная" функция для получения вектора запроса.
    Реализует логику кэширования.
    """
    if not os.path.exists(config.REQUEST_FILEPATH):
        logger.error(f"Ошибка: Файл с запросом '{config.REQUEST_FILEPATH}' не найден.")
        return None

    cache_is_valid = (
            not force_recalculate and
            os.path.exists(config.EMB_REQUEST_FILENAME_OUTPUT) and
            os.path.getmtime(config.EMB_REQUEST_FILENAME_OUTPUT) > os.path.getmtime(config.REQUEST_FILEPATH)
    )

    if cache_is_valid:
        logger.info("Запрос не изменился. Используем кэшированный вектор.")
        try:
            with open(config.EMB_REQUEST_FILENAME_OUTPUT, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                return cached_data['vector']
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Кэш-файл '{config.EMB_REQUEST_FILENAME_OUTPUT}' поврежден. Будет выполнена пересчет. Ошибка: {e}")

    logger.info("Обнаружен новый или измененный запрос. Выполняется векторизация...")
    with open(config.REQUEST_FILEPATH, 'r', encoding='utf-8') as f:
        query_text = f.read().strip()

    if not query_text:
        logger.error(f"Файл запроса '{config.REQUEST_FILEPATH}' пуст. Поиск невозможен.")
        return None
    query_vector = _build_vector(query_text, is_query=True)
    data_to_cache = {
        "request_text": query_text,
        "vector": query_vector.tolist()
    }
    with open(config.EMB_REQUEST_FILENAME_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(data_to_cache, f, ensure_ascii=False, indent=4)

    logger.info(f"Новый вектор сохранен в кэш-файл '{config.EMB_REQUEST_FILENAME_OUTPUT}'.")
    return query_vector

def create_db_vectors():
    print(f"Начинаем обработку файла '{config.CHUNKS_FILEPATH}'...")
    model = _get_embedding_model()
    with open(config.CHUNKS_FILEPATH, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f)

    with open(config.CHUNKS_FILEPATH, 'r', encoding='utf-8') as infile, \
            open(config.DATABASE, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Построение векторов"):
            if not line.strip():
                continue
            data = json.loads(line)
            text_to_embed = data.get("text")
            if text_to_embed:
                vector = model.encode(text_to_embed, convert_to_tensor=False, show_progress_bar=False)
                output_data = {
                    "vector": vector.tolist(),
                    "text": data["text"],
                    "metadata": data["metadata"]
                }
                outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

def _build_vector(text: str, is_query: bool = False):
    model = _get_embedding_model()
    if is_query:
        prompt_name = "query" if is_query else None
        vector = model.encode(text, prompt_name=prompt_name)
    else:
        vector = model.encode(text, convert_to_tensor=False)
    return vector

def cleanup_embedding_model():
    """Функция для явной очистки памяти."""
    global _model
    if _model is not None:
        del _model
        _model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Память от модели эмбеддингов очищена.")