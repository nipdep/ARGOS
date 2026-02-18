from typing import List, Optional

import pandas as pd

LIGHT_SCHEMA_TEMPLATE = """## Table: {table}
### Table description
{table_description}
### Column information
{column_information}"""
PRIMARY_KEY = """### Primary keys
{primary_key}"""
FOREIGN_KEY = """### Foreign keys
{foreign_key}"""
INDEX = """### Index
{index}"""


class LightSchema(object):
    @staticmethod
    def _to_markdown_fallback(headers: List[str], rows: List[List[object]]) -> str:
        widths = []
        for i, header in enumerate(headers):
            cell_width = len(str(header))
            for row in rows:
                cell_width = max(cell_width, len(str(row[i])))
            widths.append(cell_width)

        header_line = "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
        row_lines = [
            "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
            for row in rows
        ]
        return "\n".join([header_line, sep_line] + row_lines)

    @staticmethod
    def create_schema(
        database: str,
        table: str,
        columns: List,
        database_description: Optional[str] = "",
        table_description: Optional[str] = "",
        primary_key: Optional[List[str]] = None,
        foreign_key: Optional[List[str]] = None,
        index: Optional[List[str]] = None,
    ) -> str:
        """
        Create light schema.
        """
        column_information = LightSchema._create_column_information(columns)
        schema_items = []
        schema = LIGHT_SCHEMA_TEMPLATE.format(
            table=table,
            table_description=table_description,
            column_information=column_information,
        )
        schema_items.append(schema)
        if primary_key:
            schema_items.append(PRIMARY_KEY.format(primary_key=primary_key))
        if foreign_key:
            schema_items.append(FOREIGN_KEY.format(foreign_key=foreign_key))
        if index:
            schema_items.append(INDEX.format(index=index))
        light_schema = "\n".join(schema_items)
        return light_schema

    @staticmethod
    def _create_column_information(columns: List) -> str:
        """
        Create column information.
        """
        headers = ["column_name", "column_type", "column_description", "value_examples"]
        pd_data = {key: [] for key in headers}
        for column in columns:
            pd_data["column_name"].append(column["name"])
            pd_data["column_type"].append(column["type"])
            pd_data["column_description"].append(column.get("description", ""))
            pd_data["value_examples"].append(column["samples"])

        try:
            column_information = pd.DataFrame(pd_data)
            return column_information.to_markdown(index=False)
        except Exception:
            rows = [
                [pd_data["column_name"][i], pd_data["column_type"][i], pd_data["column_description"][i], pd_data["value_examples"][i]]
                for i in range(len(pd_data["column_name"]))
            ]
            return LightSchema._to_markdown_fallback(headers=headers, rows=rows)
