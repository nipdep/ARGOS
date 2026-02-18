from .base import ChatBIBaseException, ErrorCodeEnum


class RetrieveServiceException(ChatBIBaseException):
    @property
    def error_code(self) -> ErrorCodeEnum:
        return ErrorCodeEnum.RETRIEVE_SERVICE_ERROR

    @property
    def retryable(self) -> bool:
        return False


class RetrieveTimeoutException(ChatBIBaseException):
    @property
    def error_code(self) -> ErrorCodeEnum:
        return ErrorCodeEnum.RETRIEVE_SERVICE_TIMEOUT

    @property
    def retryable(self) -> bool:
        return True
