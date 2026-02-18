from collections import defaultdict

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from scalesql.exceptions import ExecutionServiceException
from scalesql.executions.base import (
    BaseDatabaseExecutor,
    QueryExecutionRequest,
    QueryExecutionResponse,
)


class SQLAlchemyExecutor(BaseDatabaseExecutor):
    """
    基于 SQLAlchemy 实现的数据库执行器。
    """

    def __init__(self, connection_string: str):
        """
        初始化 SQLAlchemyExecutor。

        Args:
            connection_string (str): SQLAlchemy 风格的数据库连接字符串。
            例如：'postgresql://user:password@host:port/database' 或 'sqlite:///./test.db'

        Raises:
            ExecutionServiceException: 如果提供的连接字符串无效或无法连接到数据库。
        """
        super().__init__(connection_string)
        self.engine = None
        try:
            self.engine = create_engine(self.connection_string)
            self.check_connection()
        except SQLAlchemyError as e:
            raise ExecutionServiceException(f"数据库初始化或连接失败: {str(e)}")
        except Exception as e:
            raise ExecutionServiceException(f"创建引擎时发生未知错误: {str(e)}")

    def check_connection(self) -> None:
        """
        测试与数据库的连接是否正常。

        Raises:
            ExecutionServiceException: 如果无法建立连接。
        """
        try:
            with self.engine.connect():
                pass
        except SQLAlchemyError as e:
            raise ExecutionServiceException(f"数据库连接测试失败: {str(e)}")

    def execute_query(self, query: QueryExecutionRequest) -> QueryExecutionResponse:
        """
        执行一个SQL查询，并以列式存储格式返回结果。

        Args:
            query (QueryExecutionRequest): 包含要执行的SQL查询字符串的请求对象。

        Returns:
            QueryExecutionResponse: 包含查询结果或错误信息的响应对象。
        """
        if not self.engine:
            raise ExecutionServiceException("数据库引擎未初始化。")

        try:
            with self.engine.connect() as conn:
                execution_results = conn.execute(text(query.query))
                results = defaultdict(list)
                for row in execution_results:
                    for key, value in row._mapping.items():
                        results[key].append(value)

                return QueryExecutionResponse(results=dict(results))

        except Exception as e:
            return QueryExecutionResponse(error_message=str(e))
