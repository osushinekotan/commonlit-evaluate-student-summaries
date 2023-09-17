import logging
import time
from contextlib import contextmanager

import psutil
from colorlog import ColoredFormatter


class Logger:
    def __init__(self, name: str = "", filename=None, level=logging.DEBUG, filemode="a"):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self.logger.setLevel(level)

        # Define formatter
        log_format = "[%(asctime)s] %(log_color) s[%(name)s] [%(levelname)s] - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = ColoredFormatter(
            log_format,
            datefmt=date_format,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Check if the logger already has handlers. If not, add new handlers.
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            if filename is not None:
                file_handler = logging.FileHandler(filename, mode=filemode)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def log_memory_usage(self) -> None:
        """Logs the current memory usage."""
        memory_info = psutil.virtual_memory()

        total_memory = memory_info.total / (1024**3)  # in GB
        used_memory = memory_info.used / (1024**3)  # in GB
        memory_percentage = memory_info.percent

        self.info(f"- Total Memory: {total_memory:.2f} GB 💾")
        self.info(f"- Used Memory: {used_memory:.2f} GB 💾")
        self.info(f"- Memory Usage Percentage: {memory_percentage}% 📊")

    @contextmanager
    def profile(self, target: str | None = None) -> None:
        target = self.logger.name if target is None else target
        start_time = time.time()
        self.info(f"\n Start {target} 🚀 ")
        try:
            yield
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.info(f"\n End {target} ✨")
            self.info(f"- Elapsed time: {elapsed_time:.2f} seconds ⏰")
            self.log_memory_usage()
