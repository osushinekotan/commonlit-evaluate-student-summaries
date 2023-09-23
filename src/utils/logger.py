import logging
import time
from contextlib import contextmanager

import psutil


class HydraLogger:
    def __init__(self, name: str) -> None:
        self.name = name

    def debug(self, message):
        logging.debug(message)

    def info(self, message):
        logging.info(message)

    def warning(self, message):
        logging.warning(message)

    def error(self, message):
        logging.error(message)

    def critical(self, message):
        logging.critical(message)

    def log_memory_usage(self) -> None:
        """Logs the current memory usage."""
        memory_info = psutil.virtual_memory()

        total_memory = memory_info.total / (1024**3)  # in GB
        used_memory = memory_info.used / (1024**3)  # in GB
        memory_percentage = memory_info.percent

        self.info(f"Total Memory: {total_memory:.2f} GB 💾")
        self.info(f"Used Memory: {used_memory:.2f} GB 💾")
        self.info(f"Memory Usage Percentage: {memory_percentage}% 📊")

    @contextmanager
    def profile(self, target: str | None = None) -> None:
        target = self.name if target is None else target
        start_time = time.time()
        self.info(f"# -------------- # Start {target} 🚀 # -------------- #")
        try:
            yield
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.info(f"Elapsed time: {elapsed_time:.2f} seconds ⏰")
            self.log_memory_usage()
            self.info(f"# -------------- # End {target} ✨ # -------------- #")
