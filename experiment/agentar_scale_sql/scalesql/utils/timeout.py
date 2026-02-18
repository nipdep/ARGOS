import concurrent.futures
import functools
from typing import Type, Union


def timeout(
    timeout_sec: float, exception: Union[Type[Exception], Exception] = TimeoutError
) -> callable:
    """
    超时处理装饰器

    参数:
        timeout_sec: 超时时间（秒）
        exception: 超时时抛出的异常类型或异常实例（默认TimeoutError）

    返回:
        装饰器函数

    示例:
        @timeout(5.0, exception=TimeoutError("操作超时"))
        def long_operation():
            time.sleep(10)
    """

    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> any:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_sec)
                except concurrent.futures.TimeoutError as exc:
                    # 清理未完成的 future
                    future.cancel()
                    # 构造并抛出自定义异常
                    if isinstance(exception, Exception):
                        raise exception from exc
                    raise exception(
                        f"'{func.__name__}' timed out after {timeout_sec} seconds"
                    ) from exc

        return wrapper

    return decorator
