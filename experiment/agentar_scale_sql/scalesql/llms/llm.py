from typing import Optional, Dict, List
import httpx
from langchain_openai import ChatOpenAI


def create_openai_llm(
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        request_timeout_seconds: float = 120.0,
        **kwargs,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with the specified configuration
    """
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {
        "model": model,
        "temperature": temperature,
        "max_retries": max_retries,
        "timeout": request_timeout_seconds,
        **kwargs,
    }
    if base_url:  # This will handle None or empty string
        llm_kwargs["base_url"] = base_url
    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key
    # Disable SSL verification and use bounded request timeouts so long runs do not hang forever.
    timeout_obj = httpx.Timeout(
        timeout=request_timeout_seconds,
        connect=min(10.0, request_timeout_seconds),
        read=request_timeout_seconds,
        write=min(30.0, request_timeout_seconds),
    )
    llm_kwargs["http_client"] = httpx.Client(verify=False, timeout=timeout_obj)
    return ChatOpenAI(**llm_kwargs)


def get_llm_list(configs: Dict, key: str) -> List[ChatOpenAI]:
    llm_list = []
    llm_config = configs["llm_config"]
    module_config = configs["module_configs"][key]
    for each_llm in module_config:
        if each_llm.get("thinking_budget") is not None:
            llm_list.append(
                create_openai_llm(
                    model=llm_config[each_llm["model"]]["model_name"],
                    base_url=llm_config[each_llm["model"]]["base_url"],
                    api_key=llm_config[each_llm["model"]]["api_key"],
                    temperature=each_llm["temperature"],
                    model_kwargs={
                        "extra_body": {
                            "reasoning": {
                                "max_tokens": each_llm.get("thinking_budget")
                            }
                        }
                    }
                )
            )
        elif each_llm.get("effort") is not None:
            llm_list.append(
                create_openai_llm(
                    model=llm_config[each_llm["model"]]["model_name"],
                    base_url=llm_config[each_llm["model"]]["base_url"],
                    api_key=llm_config[each_llm["model"]]["api_key"],
                    temperature=each_llm["temperature"],
                    model_kwargs={
                        "extra_body": {
                            "reasoning": {
                                "effort": each_llm.get("effort")
                            }
                        }
                    }
                )
            )
        else:
            llm_list.append(
                create_openai_llm(
                    model=llm_config[each_llm["model"]]["model_name"],
                    base_url=llm_config[each_llm["model"]]["base_url"],
                    api_key=llm_config[each_llm["model"]]["api_key"],
                    temperature=each_llm["temperature"]
                )
            )

    return llm_list
