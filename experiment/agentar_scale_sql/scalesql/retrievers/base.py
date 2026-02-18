from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Condition(BaseModel):
    field: str
    operator: Literal["=", "!=", "in", "not in"] = Field(default="in")
    value: List[Any]


class RetrieveRequest(BaseModel):
    search_query: str
    mode: Literal["text", "vector", "hybrid"]
    index_name: str
    filter_conditions: Optional[List[Condition]] = Field(default_factory=list)
    threshold: Optional[float] = Field(default=0.0)
    k: Optional[int] = Field(default=10)
    only_retrieve: Optional[bool] = Field(default=None)
    rerank_mode: Optional[Literal["bge", "gte", "rrf"]] = Field(default=None)
    trace_id: Optional[str] = Field(default=None)
    extra_args: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def to_api_payload(self) -> Dict[str, Any]:
        """Convert request to API payload format"""
        return {
            "query": self.search_query,
            "mode": self.mode,
            "collectionName": self.index_name,
            "filterConditions": [
                condition.model_dump() for condition in self.filter_conditions
            ],
            "threshold": self.threshold,
            "limit": self.k,
            "onlyRetrieve": self.only_retrieve,
            "rerankMode": self.rerank_mode,
            "traceId": self.trace_id,
        }


class RetrieveDoc(BaseModel):
    content: str
    biz_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    score: Optional[float] = Field(default=None)


class RetrieveResponse(BaseModel):
    docs: Optional[List[RetrieveDoc]] = Field(default_factory=list)
    error_message: Optional[str] = Field(default=None)


class BaseVectorStore(ABC):
    """Base class for vector store."""

    def __init__(self, index_name: str = None):
        self.index_name = index_name
        self.client = None

    @abstractmethod
    def add_documents(self, documents: List[Dict]) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents (List[Dict]): List of documents to be added.

        Returns:
            list[str]: List of IDs of the added documents.
        """
        pass

    @abstractmethod
    def search(self, retrieve_request: RetrieveRequest) -> RetrieveResponse:
        """Search for documents in the vector store.

        Args:
            retrieve_request (RetrieveRequest): Retrieve request.

        Returns:
            RetrieveResponse: Retrieve response.
        """
        pass
