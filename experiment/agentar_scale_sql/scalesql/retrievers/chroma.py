from typing import List

import chromadb

from scalesql.retrievers import RetrieveRequest, RetrieveResponse
from scalesql.retrievers.base import BaseVectorStore, RetrieveDoc


class ChromaVectorStore(BaseVectorStore):
    """
    基于 ChromaDB 实现的 VectorStore。
    """

    def __init__(self, client_path: str, index_name: str):
        """
        初始化 ChromaVectorStore。
        """
        super().__init__(index_name)
        self.collection = None
        self.client = chromadb.PersistentClient(path=client_path)

    def connect(self):
        """
        建立与 Chroma 数据库的连接，并获取或创建集合。
        """
        if self.collection is None:
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.index_name
                )
            except Exception as e:
                raise Exception(f"Error connecting to chromadb collection: {e}") from e

    def disconnect(self):
        """
        对于 ChromaDB，通常不需要显式地“断开”本地客户端。
        对于 HTTP 客户端，连接由底层 HTTP 库管理。
        """
        pass

    def add_documents(self, documents: List[RetrieveDoc]) -> List[str]:
        if self.collection is None:
            raise Exception(
                "No active chromadb collection connection. Call connect() first."
            )

        try:
            existing_count = self.collection.count()
            contents = [document.content for document in documents]
            metadatas = [document.biz_data for document in documents]
            ids = [f"doc_{i + existing_count}" for i in range(len(contents))]

            self.collection.add(documents=contents, metadatas=metadatas, ids=ids)

            print("Data successfully add to the collection!")

        except Exception as e:
            raise Exception(f"Error adding documents to chromadb collection: {e}")

        return ids

    def search(self, retrieve_request: RetrieveRequest) -> RetrieveResponse:
        try:
            where_clause = None
            if retrieve_request.filter_conditions:
                and_conditions = []
                for cond in retrieve_request.filter_conditions:
                    if cond.operator.lower() == "in":
                        op = "$in"
                    elif cond.operator.lower() == "nin":
                        op = "$nin"
                    else:
                        continue

                    and_conditions.append({cond.field: {op: cond.value}})

                if and_conditions:
                    where_clause = {"$and": and_conditions}

            results = self.collection.query(
                query_texts=[retrieve_request.search_query],
                n_results=retrieve_request.k,
                where=where_clause,
            )

            docs = []
            if results and results.get("documents"):
                result_documents = results["documents"][0]
                result_metadatas = results["metadatas"][0]
                result_distances = results["distances"][0]

                for document, metadata, distance in zip(
                        result_documents, result_metadatas, result_distances
                ):
                    # 只保留那些满足质量阈值（距离足够近）的结果
                    if distance < retrieve_request.threshold:
                        docs.append(
                            RetrieveDoc(content=document, biz_data=metadata)
                        )

            return RetrieveResponse(docs=docs)

        except Exception as e:
            error = str(e)
            return RetrieveResponse(error_message=error)
