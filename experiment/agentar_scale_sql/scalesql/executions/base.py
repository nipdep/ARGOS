from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryExecutionRequest(BaseModel):
    """Query execution request model."""

    query: str
    datasource_id: Optional[str] = Field(default=None)
    table_id: Optional[List[str]] = Field(default_factory=list)
    dialect: Optional[str] = Field(default=None)
    trace_id: Optional[str] = Field(default=None)
    extra_args: Optional[Dict[str, Any]] = Field(default_factory=dict)


class QueryExecutionResponse(BaseModel):
    """Query execution response model."""

    results: Optional[Dict[str, Any]] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)


class BaseDatabaseExecutor(ABC):
    """
    数据库执行的抽象基类。
    定义了数据库操作的基本接口，但不涉及具体的数据库实现细节。
    """

    def __init__(self, connection_string: str = None):
        """
        初始化基类，接收数据库连接字符串。
        """
        self.connection_string = connection_string

    @abstractmethod
    def execute_query(
        self, execute_request: QueryExecutionRequest
    ) -> QueryExecutionResponse:
        """
        抽象方法：执行数据库查询。

        Args:
            execute_request (QueryExecutionRequest): 执行查询的请求体。

        Returns:
            QueryExecutionResponse: 查询结果。
        """
        pass
