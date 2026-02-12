import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def normalize_name(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text[0] in {'"', "'", "`", "["} and text[-1] in {'"', "'", "`", "]"}:
        text = text[1:-1]
    return text.strip().lower()


def quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


@lru_cache(maxsize=None)
def load_local_module(module_name: str, filename: str) -> ModuleType:
    base_dir = Path(__file__).resolve().parent
    module_path = base_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
