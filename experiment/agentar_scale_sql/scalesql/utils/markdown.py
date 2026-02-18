from typing import Dict


def dict_to_markdown(data: Dict) -> str:
    if not data:
        return ""

    # 获取表头
    headers = list(data.keys())

    # 构建表头行
    header_row = "| " + " | ".join(headers) + " |"
    # 构建分隔行
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"

    # 构建数据行
    rows = []
    for i in range(len(next(iter(data.values())))):
        row = []
        for key in headers:
            value = data[key][i]
            formatted_value = str(value)
            row.append(formatted_value)
        rows.append("| " + " | ".join(row) + " |")

    # 组合所有行
    return "\n".join([header_row, separator] + rows)
