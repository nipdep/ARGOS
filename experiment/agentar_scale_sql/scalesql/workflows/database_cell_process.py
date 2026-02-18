import argparse
import logging
import os
from uuid import uuid4
from scalesql.utils.utils import get_default_device, get_cursor_from_path
from scalesql.utils import setup_logging
import chromadb
import yaml
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

setup_logging()


class ChromaWriter:
    def __init__(
            self,
            database_folder,
            dataset_cell_chroma_path,
            embedding_model_path,
            device="mps",
            batch_size=1000,
            max_str_len=256,
            embedding_model="all-MiniLM-L6-v2",
    ):
        self.database_folder = database_folder
        self.dataset_cell_chroma_path = dataset_cell_chroma_path
        self.embedding_model_path = embedding_model_path

        if not os.path.exists(embedding_model_path):
            logging.info(
                f"本地模型路径 {embedding_model_path} 不存在，正在从 Hugging Face 下载 {embedding_model} ..."
            )
            model = SentenceTransformer(embedding_model)
            self.embedding_model_path = (
                    model.save(embedding_model_path) or embedding_model_path
            )
        else:
            self.embedding_model_path = embedding_model_path

        self.device = device
        self.batch_size = batch_size
        self.max_str_len = max_str_len
        self.skip_keywords = [
            "_id",
            " id",
            "url",
            "email",
            "web",
            "time",
            "date",
            "address",
        ]

    def process_single_db(self, collection_name):
        client = chromadb.PersistentClient(path=self.dataset_cell_chroma_path)
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_path,
            device=self.device,
        )
        # 只 get 已存在的 collection，不再 create
        collection = client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
        db_path = f"{self.database_folder}/{collection_name}/{collection_name}.sqlite"
        logging.info(f"Processing {db_path}")

        cursor = get_cursor_from_path(db_path)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        num_cells = 0
        for table in tables:
            values, metadatas, ids = [], [], []
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info(`{table_name}`);")
            columns = cursor.fetchall()
            primary_keys = [col[1].lower() for col in columns if col[5] > 0]
            string_columns = [
                col[1]
                for col in columns
                if "text" in col[2].lower()
            ]

            for col_name in string_columns:
                col_lower = col_name.lower()
                if (
                        col_lower in primary_keys
                        or any(keyword in col_lower for keyword in self.skip_keywords)
                        or col_lower.endswith("id")
                ):
                    logging.info(
                        f"Skipping column {col_name} in table {table_name} due to filter."
                    )
                    continue
                query = f"SELECT DISTINCT `{col_name}` FROM `{table_name}` WHERE `{col_name}` IS NOT NULL"

                cursor.execute(query)
                rows = cursor.fetchall()

                filtered_values = [
                    row[0]
                    for row in rows
                    if isinstance(row[0], str) and len(row[0]) <= self.max_str_len
                ]

                metadatas.extend(
                    {"table": table_name, "column": col_name}
                    for _ in range(len(filtered_values))
                )
                values.extend(filtered_values)
                for i in range(len(filtered_values)):
                    ids.append(str(uuid4()))
            total_batches = (
                (len(values) + self.batch_size - 1) // self.batch_size if values else 0
            )
            for i in range(0, len(values), self.batch_size):
                batch_values = values[i: i + self.batch_size]
                batch_metadatas = metadatas[i: i + self.batch_size]
                batch_ids = ids[i: i + self.batch_size]
                if batch_values:
                    collection.add(
                        documents=list(batch_values),
                        metadatas=list(batch_metadatas),
                        ids=list(batch_ids),
                    )
                    logging.info(
                        f"[{collection_name}] Table: {table_name} | Batch {i // self.batch_size + 1}/{total_batches} 已写入 {min(i + self.batch_size, len(values))}/{len(values)}"
                    )
            num_cells += len(values)
        logging.info(f"[Success] 共写入 {num_cells} 条字符串值到 ChromaDB 集合 '{collection_name}' 中。")
        # conn.close()

    def process_db(self, collections):
        client = chromadb.PersistentClient(path=self.dataset_cell_chroma_path)
        exist_collections = [col.name for col in client.list_collections()]

        # 1. 先顺序创建所有 collection
        for collection_name in collections:
            if collection_name in exist_collections:
                try:
                    client.delete_collection(collection_name)
                    logging.info(f"Deleted existing collection: {collection_name}")
                except Exception as e:
                    logging.warning(f"Delete collection error: {e}")

            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_path,
                device=self.device,
            )
            try:
                client.create_collection(
                    name=collection_name, embedding_function=embedding_function
                )
                logging.info(f"Created collection: {collection_name}")
            except Exception as e:
                logging.warning(f"Create collection error: {e}")

        # 2. 顺序处理每一个数据库
        logging.info("Starting to process databases sequentially...")
        for collection_name in collections:
            try:
                self.process_single_db(collection_name)
                logging.info(f"[Success] 集合 '{collection_name}' 处理成功。")
            except Exception as e:
                # 记录详细的 traceback 信息会更有帮助
                logging.error(f"[Error] 集合 '{collection_name}' 处理失败，错误信息：{e}", exc_info=True)


def ChromaWriteMain(database_folder, dataset_cell_chroma_path, embedding_model_path):
    def return_dbs_in_dataset(db_file_folder):
        """返回数据集文件夹下所有数据库名"""
        return [
            f
            for f in os.listdir(db_file_folder)
            if os.path.isdir(os.path.join(db_file_folder, f))
        ]

    collections = return_dbs_in_dataset(database_folder)
    logging.info(f"有 {len(collections)} 个数据库需要处理: {collections}")
    chroma_writer = ChromaWriter(
        database_folder=database_folder,
        dataset_cell_chroma_path=dataset_cell_chroma_path,
        embedding_model_path=embedding_model_path,
        device=get_default_device()
    )
    chroma_writer.process_db(collections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        required=True,
        type=str,
        choices=["dev", "test", "train"],
        default="test"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="scalesql/workflows/config/pipeline_config.yaml"
    )
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    for key, value in vars(args).items():
        if value is not None:
            configs[key] = value

    logging.info(f"configs:\n{configs}")

    database_folder = configs["dataset_folder"] + "/{}_databases".format(configs["evaluation_type"])
    if os.path.isdir("/tmp"):
        base_path = "/tmp"
    else:
        base_path = "."
    dataset_cell_chroma_path = os.path.join(
        base_path,
        "scalesql/chroma/bird_{}".format(configs["evaluation_type"])
    )
    embedding_model_path = "./scalesql/model/all-MiniLM-L6-v2"

    ChromaWriteMain(
        database_folder=database_folder,
        dataset_cell_chroma_path=dataset_cell_chroma_path,
        embedding_model_path=embedding_model_path
    )
