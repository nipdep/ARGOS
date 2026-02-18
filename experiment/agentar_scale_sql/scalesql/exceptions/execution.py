from .base import ChatBIBaseException, ErrorCodeEnum


class ExecutionServiceException(ChatBIBaseException):
    @property
    def error_code(self) -> ErrorCodeEnum:
        return ErrorCodeEnum.EXECUTION_SERVICE_ERROR

    @property
    def retryable(self) -> bool:
        return False


class ExecutionTimeoutException(ChatBIBaseException):
    @property
    def error_code(self) -> ErrorCodeEnum:
        return ErrorCodeEnum.EXECUTION_SERVICE_TIMEOUT

    @property
    def retryable(self) -> bool:
        return True
