import logging
from llama_cpp import Llama

from . import config

logger = logging.getLogger(__name__)

_llm_instance: Llama | None = None


def get_llm() -> Llama:
    global _llm_instance
    if _llm_instance is None:
        logging.info(f"Loading model from {config.ASN_MODEL_PATH}")
        try:
            _llm_instance = Llama(
                model_path=config.ASN_MODEL_PATH,
                n_ctx=config.ANS_NCTX,
                n_threads=config.ANS_NTHREADS,
                n_gpu_layers=config.ANS_NGPU,
                verbose=False
            )
            logging.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from '{config.ASN_MODEL_PATH}' : {e}")
            raise
    return _llm_instance


def send_prompt(prompt: str) -> dict:
    try:
        llm = get_llm()
        logging.info("Sending request.txt to LLM...")
        response = llm(
            prompt=prompt,
            max_tokens=config.ANS_MAX_TOKENS,
            temperature=config.ANS_TEMPERATURE,
            stop=config.ANS_STOP_SEQUENCES,
            echo=False
        )
        logging.info("Received response from LLM.")
        return response
    except Exception as e:
        logger.exception(f"Error during generating response : {e}")
        raise