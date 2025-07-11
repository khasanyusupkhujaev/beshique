import logging
from datetime import datetime

class Logger:
    def __init__(self, filename='baby_monitor.log'):  # Adjust path for Pi
        self.logger = logging.getLogger('BabyMonitor')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)
