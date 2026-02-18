import enum
from abc import ABC, abstractmethod


class ErrorCodeEnum(enum.Enum):
    # system
    ILLEGAL_PARAM = 2001
    AGENT_CONFIG_ERROR = 2002
    MODEL_CONFIG_ERROR = 2003
    AGENT_BUILD_ERROR = 2004
    AUTH_ERROR = 2005
    # llm related
    LLM_SERVICE_ERROR = 4001
    LLM_GENERATION_PARSE_ERROR = 4002
    LLM_LENGTH_ERROR = 4003
    # retriever related
    RETRIEVE_SERVICE_ERROR = 4004
    RETRIEVE_SERVICE_TIMEOUT = 4005
    # execution related
    EXECUTION_SERVICE_ERROR = 4006
    EXECUTION_SERVICE_TIMEOUT = 4007
    # white box related
    WHITE_BOX_SERVICE_ERROR = 4008
    WHITE_BOX_SERVICE_TIMEOUT = 4009
    # reserved
    SERVER_INIT_ERROR = 9996
    INTERNAL_ERROR = 9997
    RUNTIME_ERROR = 9998
    UNKNOWN_ERROR = 9999


class ChatBIBaseException(Exception, ABC):
    @property
    def need_attention(self) -> bool:
        return False

    @property
    @abstractmethod
    def error_code(self) -> ErrorCodeEnum:
        raise NotImplementedError(f"{type(self)}.error_code not implemented.")

    @property
    @abstractmethod
    def retryable(self) -> bool:
        raise NotImplementedError(f"{type(self)}.retryable not implemented.")
