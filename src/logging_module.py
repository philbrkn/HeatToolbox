# logging_module.py
import logging


class LoggingModule:
    def __init__(self, log_level=logging.INFO):
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)

    def log(self, message):
        self.logger.info(message)
