class BaseDetector:
    def __init__(self):
        self.is_active = True
        self.last_detection_time = None

    def process(self, frame):
        raise NotImplementedError

    def toggle(self):
        self.is_active = not self.is_active
