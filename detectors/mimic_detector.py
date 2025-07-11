from .base_detector import BaseDetector

class MimicDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        
    def process(self, frame):
        if not self.is_active:
            return None
        # Add mimic detection logic here
        return None  # placeholder
