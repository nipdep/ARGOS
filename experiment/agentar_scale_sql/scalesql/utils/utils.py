import json
import os

import logging
from typing import List, Optional
import pandas as pd
import platform
import sqlite3
from functools import lru_cache

try:
    import torch
except ModuleNotFoundError:
    torch = None

from scalesql.executions import QueryExecutionRequest
from scalesql.executions.sqlalchemy import SQLAlchemyExecutor
from scalesql.utils.logging import setup_logging

setup_logging()

worker_cache = {
    "uris": {},
    "connections": {}
}


@lru_cache(maxsize=16)
def get_worker_db_uri(db_path: str) -> str:
    if db_path not in worker_cache["uris"]:
        source_db_uri = f'file:{db_path}?mode=ro'
        mem_db_uri = 'file::memory:'

        try:
            source_conn = sqlite3.connect(source_db_uri, uri=True, check_same_thread=False)
            mem_conn = sqlite3.connect(mem_db_uri, uri=True, check_same_thread=False)

            source_conn.backup(mem_conn)
            source_conn.close()

            worker_cache["uris"][db_path] = mem_db_uri
            worker_cache["connections"][db_path] = mem_conn

            logging.info(f"[Worker PID: {os.getpid()}] Successfully loaded '{db_path}' into private memory.")

        except Exception as e:
            logging.error(f"[Worker PID: {os.getpid()}] Failed to load DB {db_path}: {e}")
            return f'file:{db_path}?mode=ro'

    return worker_cache["connections"][db_path]


@lru_cache(maxsize=16)
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

        logging.info(f"Loading SQLite database into memory from: {sqlite_path}")

        disk_conn = sqlite3.connect(f'file:{sqlite_path}?mode=ro', uri=True, check_same_thread=False)
        disk_conn.text_factory = lambda b: b.decode(errors="ignore")

        memory_conn = sqlite3.connect(':memory:', check_same_thread=False)
        memory_conn.text_factory = lambda b: b.decode(errors="ignore")

        disk_conn.backup(memory_conn)
        disk_conn.close()

        cursor = memory_conn.cursor()
        logging.info("Database successfully loaded into memory.")
        return cursor

    except Exception as e:
        logging.info(f"Failed to load database {sqlite_path} into memory")
        raise e


def get_default_device():
    if torch is not None and torch.cuda.is_available():
        return "cuda"

    if torch is not None and platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def read_json(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件 '{filename}' 不存在")

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_or_append_json(data, filename, overwrite=False, indent=2, ensure_ascii=False):
    try:
        # 1. 路径不存在时自动创建
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # 2. 文件不存在，直接写入新文件
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            logging.info(f"✅ 文件 {filename} 已创建并写入数据。")
            return

        # 3. 文件存在且需要覆盖
        if overwrite:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            logging.info(f"✅ 文件 {filename} 已被覆盖。")
            return

        # 4. 文件存在且不允许覆盖，追加到已有内容后面，保证为list
        with open(filename, "r+", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except Exception as e:
                raise RuntimeError(f"读取原文件 {filename} 失败: {e}")

            # 如果已有内容不是列表，转为列表
            if not isinstance(existing_data, list):
                existing_data = [existing_data]

            # 判断追加内容是append还是extend
            if isinstance(data, list):
                existing_data.extend(data)
            else:
                existing_data.append(data)

            f.seek(0)
            json.dump(existing_data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.truncate()
        logging.info(f"✅ 数据已追加到文件 {filename}。")
    except Exception as e:
        logging.info(f"❌ 写入 JSON 文件 {filename} 失败: {e}")
        raise


def display_matched_contents(sequenced_database_values: List[dict[str, str]]):
    """
    for displaying matched_contents and similar_questions.
    """
    ret = ""
    if len(sequenced_database_values) > 0:
        for value in sequenced_database_values:
            table, column, content = value.get("table"), value.get("column"), value.get("content")
            if ' ' in column:
                column = f'`{column}`'

            ret += f'- {table}.{column}: {content}\n'
    return ret


def display_similar_questions(similar_questions: List[dict[str, str]]):
    """
    for displaying matched_contents and similar_questions.
    """
    ret = ""
    if len(similar_questions) > 0:
        for i, value in enumerate(similar_questions):
            ret += f"## Example {i}:\n"
            for k, v in value.items():
                ret += f"{k}: {v}\n"
            ret += "\n"
    return ret


def display_execution_result(results):
    if not results:
        return "(no rows)"

    pd_data = {}

    for column, values in results.items():
        pd_data[column] = values[:5]
    column_information = pd.DataFrame(pd_data)
    try:
        return column_information.to_markdown(index=False)
    except Exception:
        headers = list(pd_data.keys())
        rows = list(zip(*[pd_data[h] for h in headers])) if headers else []
        widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        header_line = "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
        row_lines = [
            "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
            for row in rows
        ]
        return "\n".join([header_line, sep_line] + row_lines)


def _execute_candidate_sql(
        sql: str,
        db_executor: Optional[SQLAlchemyExecutor] = None,
        access_controller=None,
        role: str = "public",
):
    if access_controller is not None:
        return access_controller.execute_query(role=role, sql_query=sql)
    if db_executor is None:
        raise ValueError("db_executor is required when access_controller is not provided")
    return db_executor.execute_query(QueryExecutionRequest(query=sql))


def display_for_selection(sql_candidates, db_path, db, role: str = "public", access_controller=None):
    db_path = db_path.format(db=db)
    db_executor = None if access_controller is not None else SQLAlchemyExecutor(
        connection_string=f"sqlite:///{db_path}"
    )

    result = ""
    for i, sql in enumerate(sql_candidates):
        execution_response = _execute_candidate_sql(
            sql=sql,
            db_executor=db_executor,
            access_controller=access_controller,
            role=role,
        )
        if execution_response.error_message is None and execution_response.results is not None:
            result += f"Candidate {i}:\n{sql}\nFive rows from the database execution results: \n{display_execution_result(execution_response.results)}\n\n"

    return result


def display_for_merge(sql_candidates, db_path, db, role: str = "public", access_controller=None):
    db_path = db_path.format(db=db)
    db_executor = None if access_controller is not None else SQLAlchemyExecutor(
        connection_string=f"sqlite:///{db_path}"
    )

    result = ""
    for i, sql in enumerate(sql_candidates):
        execution_response = _execute_candidate_sql(
            sql=sql,
            db_executor=db_executor,
            access_controller=access_controller,
            role=role,
        )
        if execution_response.error_message is None and execution_response.results is not None:
            result += f"Draft SQL {i}:\n{sql}\nFive rows from the database execution results: \n{display_execution_result(execution_response.results)}\n\n"

    return result
