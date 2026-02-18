import logging
from scalesql.retrievers.chroma import ChromaVectorStore, RetrieveRequest
from scalesql.utils import setup_logging


setup_logging()


class DatabaseCellRetrieval:
    def __init__(self, database_literals, search_client, collection_name):
        self.database_literals = database_literals
        self.search_client = ChromaVectorStore(
            client_path=search_client, index_name=collection_name
        )
        self.search_client.connect()
        self.retrieval_results = []

    def retrieve(self, threshold=0.8, k=5):
        results_set = set()
        for value in self.database_literals:
            if not isinstance(value, str) or value.isdigit():
                continue
            request = RetrieveRequest(
                search_query=value, mode="text", threshold=threshold, k=k, index_name=""
            )
            responses = self.search_client.search(request)
            contents = [doc.content for doc in responses.docs if doc.content]
            metadatas = [doc.biz_data for doc in responses.docs if doc.biz_data]
            content_meta_pairs = [
                (content, metadata) for content, metadata in zip(contents, metadatas)
            ]

            for content, metadata in content_meta_pairs:
                set_key = "{}_{}_{}".format(metadata["table"], metadata["column"], content)
                if set_key in results_set:
                    continue
                results_set.add(set_key)
                self.retrieval_results.append(
                    {
                        "table": metadata["table"],
                        "column": metadata["column"],
                        "content": content,
                    }
                )
        return self.retrieval_results


def skeleton_retrieve(
        skeleton_client_path: str,
        skeleton_collection_name: str,
        question_skeleton: str,
        threshold=1.5,
        k=15,
):
    skeleton_chroma = ChromaVectorStore(
        client_path=skeleton_client_path, index_name=skeleton_collection_name
    )
    skeleton_chroma.connect()
    request = RetrieveRequest(
        search_query=question_skeleton,
        mode="text",
        threshold=threshold,
        k=k,
        index_name="",
    )
    retrieval_results = skeleton_chroma.search(request)
    return_results = []
    for result in retrieval_results.docs:
        question = result.biz_data.get("question")
        sql = result.biz_data.get("sql")
        evidence = result.biz_data.get("evidence")
        res = "Question: {}\nEvidence: {}\nSQL: {}".format(question, evidence, sql)
        return_results.append(res)

    logging.info(f"Retrieved {len(return_results)} skeleton examples.")
    return return_results
