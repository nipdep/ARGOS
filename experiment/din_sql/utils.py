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
