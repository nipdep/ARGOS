import json
import logging
import argparse
import os, shutil
import sqlite3
import yaml
from func_timeout import func_set_timeout, FunctionTimedOut
from pathlib import Path
from scalesql.utils import setup_logging
from scalesql.utils.utils import get_cursor_from_path

setup_logging()


# execute predicted sql with a long time limitation (for buiding content index)
@func_set_timeout(3600)
def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()


def remove_contents_of_a_folder(index_path):
    # if index_path does not exist, then create it
    os.makedirs(index_path, exist_ok=True)
    # remove files in index_path
    for filename in os.listdir(index_path):
        file_path = os.path.join(index_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.info('Failed to delete %s. Reason: %s' % (file_path, e))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_content_index(db_file_path, index_path):
    cursor = get_cursor_from_path(db_file_path)

    results = execute_sql(cursor, "SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [result[0] for result in results]

    all_column_contents = []
    for table_name in table_names:
        # skip SQLite system table: sqlite_sequence
        if table_name == "sqlite_sequence":
            continue
        results = execute_sql(cursor, f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}')")
        column_names_in_one_table = [result[0] for result in results]
        for column_name in column_names_in_one_table:
            try:
                logging.info(f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL;")
                results = execute_sql(cursor,
                                      f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL;")
                column_contents = [result[0] for result in results if
                                   isinstance(result[0], str) and not is_number(result[0])]

                for c_id, column_content in enumerate(column_contents):
                    # remove empty and extremely-long contents
                    if len(column_content) != 0 and len(column_content) <= 40:
                        all_column_contents.append(
                            {
                                "id": "{}-**-{}-**-{}".format(table_name, column_name, c_id),  # .lower()
                                "contents": column_content
                            }
                        )
            except Exception as e:
                logging.info(str(e))

    os.makedirs('./data/temp_db_index', exist_ok=True)

    with open("./data/temp_db_index/contents.json", "w") as f:
        f.write(json.dumps(all_column_contents, indent=2, ensure_ascii=True))

    # Building a BM25 Index (Direct Java Implementation), see https://github.com/castorini/pyserini/blob/master/docs/usage-index.md
    cmd = f'python -m pyserini.index.lucene --collection JsonCollection --input ./data/temp_db_index --index "{index_path}" --generator DefaultLuceneDocumentGenerator --threads 16 --storePositions --storeDocvectors --storeRaw'

    d = os.system(cmd)
    logging.info(d)
    os.remove("./data/temp_db_index/contents.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        required=True,
        type=str,
        choices=["dev", "test"],
        default="dev",
        help="评估测试类型，支持 'dev' 或 'test'",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="scalesql/workflows/config/pipeline_config.yaml",
        help="YAML 配置文件路径",
    )
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    for key, value in vars(args).items():
        if value is not None:
            configs[key] = value

    logging.info(f"configs:\n{configs}")

    database_folder = configs["dataset_folder"] + "/{}_databases".format(configs["evaluation_type"])

    dataset_info = {
        # BIRD dev
        "bird_dev": {"db_path": f"{database_folder}",
                     "index_path_prefix": f"{configs['dataset_folder']}/db_contents_index"},
    }

    for dataset_name in dataset_info:
        logging.info(dataset_name)
        db_path = dataset_info[dataset_name]["db_path"]
        index_path_prefix = dataset_info[dataset_name]["index_path_prefix"]
        remove_contents_of_a_folder(index_path_prefix)
        # build content index
        db_ids = os.listdir(db_path)
        # db_ids = ["the_table's_domain_appears_to_be_related_to_demographic_and_employment_data"]
        for db_id in db_ids:
            db_file_path = os.path.join(db_path, db_id, db_id + ".sqlite")
            if os.path.exists(db_file_path) and os.path.isfile(db_file_path):
                logging.info(f"The file '{db_file_path}' exists.")
                build_content_index(
                    db_file_path,
                    os.path.join(index_path_prefix, db_id)
                )
            else:
                logging.info(f"The file '{db_file_path}' does not exist.")
