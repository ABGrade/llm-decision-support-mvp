import os
from dotenv import load_dotenv

load_dotenv()

""" Общие """
REQUEST_FILEPATH = os.getenv("REQUEST_FILEPATH", "./data/request.txt")
ANSWER_FILEPATH = os.getenv("ANSWER_FILEPATH", "./data/answer.txt")
DATABASE = os.getenv("DATABASE", "./data/satellite_chunks_output.jsonl")
FOUND_RELEVANT_FILEPATH = os.getenv("FOUND_RELEVANT_FILEPATH", "./data/retrieved_results.json")
PROMPT = os.getenv("PROMPT")

""" Конфигурация логгера """
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

""" Конфигурация модели для формирования ответа """
ASN_MODEL_PATH = os.getenv("ASN_MODEL_PATH", "./model/YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf")
ANS_MAX_TOKENS = int(os.getenv("ANS_MAX_TOKENS", 32768))
ANS_TEMPERATURE = float(os.getenv("ANS_TEMPERATURE", 0.1))
ANS_NCTX = int(os.getenv("ANS_NCTX", 8192))
ANS_NTHREADS = int(os.getenv("ANS_NTHREADS", 6))
ANS_NGPU = int(os.getenv("ANS_NGPU", -1))
ANS_STOP_SEQUENCES = ["</s>", "[/INST]"]

""" Конфигурация модели для эмбединга """
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
EMB_DIMENSIONS = int(os.getenv("EMB_DIMENSIONS", 2560))
CHUNKS_FILEPATH = os.getenv("CHUNKS_FILEPATH", "./data/chunks/satellite_chunks.jsonl")
EMB_REQUEST_FILENAME_OUTPUT = os.getenv("EMB_REQUEST_FILENAME_OUTPUT", "./data/request_output.jsonl")

""" Конфигурация клиента qdrant """
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "satellite")
TOP_K = int(os.getenv("TOP_K", 5))