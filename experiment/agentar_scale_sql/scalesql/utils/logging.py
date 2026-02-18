import logging
import sys


def setup_logging(log_level=logging.INFO, log_file="app.log"):
    """一个更专业的日志配置函数"""

    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 创建一个 Formatter
    log_format = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
    )

    # 1. 配置控制台输出 (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    # 可以为不同的 handler 设置不同的日志级别
    console_handler.setLevel(logging.INFO)

    # 2. 配置文件输出 (FileHandler)
    # 'a' 表示追加模式
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_format)
    # 文件中可以记录更详细的 DEBUG 信息
    file_handler.setLevel(logging.DEBUG)

    # 将 handlers 添加到根 logger
    # 防止重复添加 handler
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
