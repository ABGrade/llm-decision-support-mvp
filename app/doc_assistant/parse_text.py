from . import config

def create_multi_sentence_prompt(sentences, initial_request) -> str:
    if not sentences:
        return ""

    prompt_header = config.PROMPT
    request = f"Запрос: {initial_request}\n"
    sentences = "Предложения для анализа:\n" + "\n".join(sent for sent in sentences)
    prompt_footer = "[/INST]"

    full_prompt = f"{prompt_header}{request}{sentences}\n{prompt_footer}"
    return full_prompt