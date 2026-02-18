import os

from dotenv import load_dotenv


def read_env() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(base_dir, '.env')
    print(env_path)
    load_dotenv(dotenv_path=env_path)
    return {
        "GEMINI": {
            "model_name": os.getenv("GEMINI", ""),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "base_url": os.getenv("LLM_BASE_URL", ""),
        },
        "GEMINI-FLASH": {
            "model_name": os.getenv("GEMINI_FLASH", ""),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "base_url": os.getenv("LLM_BASE_URL", ""),
        },
        "GPT-5": {
            "model_name": os.getenv("GPT-5", ""),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "base_url": os.getenv("LLM_BASE_URL", ""),
        }
    }
