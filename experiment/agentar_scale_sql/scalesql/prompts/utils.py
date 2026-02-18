import os
from functools import lru_cache
from typing import List

import yaml
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


def get_prompt_dir():
    return os.path.dirname(__file__)


@lru_cache(maxsize=1000)
def get_prompt_yml_content(fn: str):
    with open(os.path.join(get_prompt_dir(), fn), "r") as fh:
        data = yaml.safe_load(fh)
    return data


_prompt_cache: dict[str, ChatPromptTemplate] = {}


def get_prompt(module: str, version: str, input_keys: List[str]) -> ChatPromptTemplate:
    """
    Get ChatPromptTemplate by modules and the corresponding version. Returns cached instance if available.
    """
    cache_key = f"{module}_{version}_{input_keys}"
    if cache_key in _prompt_cache:
        return _prompt_cache[cache_key]

    messages = []

    prompt_content = get_prompt_yml_content(f"{module}.yaml")[version]
    if prompt_content.get("system_prompt"):
        system_prompt = prompt_content.get("system_prompt").replace(
            "{input_keys}", str(input_keys)
        )
        messages.append(SystemMessagePromptTemplate.from_template(system_prompt))
    information_str = "\n\n".join(f"# {key}\n{{{key}}}" for key in input_keys)
    user_prompt = prompt_content.get("user_prompt").replace(
        "{information_str}", information_str
    )
    messages.append(HumanMessagePromptTemplate.from_template(user_prompt))
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    _prompt_cache[cache_key] = chat_prompt
    return chat_prompt
