class Config:
    def __init__(self):
        # Resolution optimized for Raspberry Pi streaming
        self.resolution = (640, 480)  # Lowered from (640, 480) for performance
        self.enable_night_mode = True
        self.detection_interval = 0.1
        # Add more configuration parameters as needed