import logging
import os, json

from . import config
from . import parse_text
from . import llm_service, database_manager

logger = logging.getLogger(__name__)

def save_retrieved_results(relevants):
    """
    Сохраняет найденные релевантные фрагменты в JSON-файл, исключая вектор.
    """
    results_to_save = []
    for i, hit in enumerate(relevants):
        output_data = {
            "id": i + 1,
            "score": f"{hit.score:.4f}",
            "payload": hit.payload
        }
        results_to_save.append(output_data)
    with open(config.FOUND_RELEVANT_FILEPATH, "w", encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)
    logger.info(f"Найденные результаты сохранены в файл: {config.FOUND_RELEVANT_FILEPATH}")


def proceed():
    relevant = database_manager.search_in_db()
    if not relevant:
        logger.error("В базе данных не найдено релевантных документов. Завершение работы.")
        return None
    save_retrieved_results(relevant)
    try:
        texts_for_prompt = []
        sources = {}
        for hit in relevant:
            payload = hit.payload
            text = payload.get('text', '')
            texts_for_prompt.append(text)
            metadata = payload.get("metadata", {})
            source_file = metadata.get('source')
            page_num = metadata.get('page')
            if source_file and page_num:
                source_key = (source_file, page_num)
                sources[source_key] = f"{source_file} (страница {page_num})"
        with open(config.EMB_REQUEST_FILENAME_OUTPUT, 'r', encoding='utf-8') as f:
            data = json.load(f)
            request_text = data['request_text']
        full_prompt = parse_text.create_multi_sentence_prompt(texts_for_prompt, request_text)
    except Exception as e:
        logger.error(f"Ошибка при подготовке данных для LLM: {e}")
        raise e
    try:
        llm_response = llm_service.send_prompt(full_prompt)
    except Exception as e:
        logger.error(f"Ошибка при обращении к LLM: {e}")
        raise e
    raw_text = llm_response.get('choices', [{}])[0].get('text', 'Не удалось получить ответ от LLM.').strip()
    sources_line = "\n\nИсточники:\n- " + "\n- ".join(sorted(list(sources.values()))) if sources else ""
    final_answer_with_sources = raw_text + sources_line
    with open(config.ANSWER_FILEPATH, "w", encoding='utf-8') as f:
        f.write(final_answer_with_sources)
    logger.info(f"Финальный ответ с источниками сохранен в: {config.ANSWER_FILEPATH}")