import os
import argparse
import json
import logging
from scalesql.utils.utils import get_default_device
from scalesql.utils import setup_logging
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

setup_logging()


class TrainSkeletonWriter:
    def __init__(
            self,
            skeleton_file_path,
            chroma_client_path,
            collection_name,
            embedding_model_path,
            device="mps",
            batch_size=1000
    ):
        self.skeleton_file_path = skeleton_file_path
        self.chroma_client_path = chroma_client_path
        self.collection_name = collection_name
        self.embedding_model_path = embedding_model_path

        if not os.path.exists(embedding_model_path):
            raise Exception(
                f"本地模型路径 {embedding_model_path} 不存在，参考README从modelscope上下载"
            )
        else:
            self.embedding_model_path = embedding_model_path

        self.device = device
        self.batch_size = batch_size

    def write(self):
        # 读取 skeleton 文件
        with open(self.skeleton_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        client = chromadb.PersistentClient(path=self.chroma_client_path)
        try:
            client.delete_collection(self.collection_name)
        except Exception as e:
            logging.warning(f"Delete collection error: {e}")
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_path,
            device=self.device,
        )
        collection = client.create_collection(
            name=self.collection_name, embedding_function=embedding_function
        )

        docs, metadatas, ids = [], [], []
        for idx, item in enumerate(data):
            docs.append(item["skeleton"])
            metadatas.append(
                {
                    "question": item.get("question", ""),
                    "sql": item.get("sql", ""),
                    "evidence": item.get("evidence", ""),
                    "db": item.get("db", ""),
                    "id": item.get("id", idx),
                }
            )
            ids.append(f"id_{idx}")

        # 分批写入
        for i in range(0, len(docs), self.batch_size):
            batch_docs = docs[i: i + self.batch_size]
            batch_metadatas = metadatas[i: i + self.batch_size]
            batch_ids = ids[i: i + self.batch_size]
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
        logging.info(
            f"[Process] 共写入 {len(docs)} 条 skeleton 到 ChromaDB 集合 '{self.collection_name}' 中。"
        )


def main():
    skeleton_file_path = "./scalesql/dataset/bird_train.json"
    if os.path.isdir("/tmp"):
        base_path = "/tmp"
    else:
        base_path = "."
    chroma_client_path = os.path.join(
        base_path,
        "scalesql/chroma/bird_train_skeleton/"
    )
    collection_name = "bird_train_skeleton"
    embedding_model_path = "./scalesql/model/all-MiniLM-L6-v2"
    writer = TrainSkeletonWriter(
        skeleton_file_path=skeleton_file_path,
        chroma_client_path=chroma_client_path,
        collection_name=collection_name,
        embedding_model_path=embedding_model_path,
        device=get_default_device()
    )
    writer.write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()
