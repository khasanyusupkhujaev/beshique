from picamera2 import Picamera2
import cv2
from .frame_processor import FrameProcessor

class CameraManager:
    def __init__(self, resolution=(640, 480)):
        # Initialize the Raspberry Pi Camera
        self.resolution = resolution
        self.camera = Picamera2()
        # Configure the camera with the specified resolution
        config = self.camera.create_video_configuration(main={"size": resolution})
        self.camera.configure(config)
        # Start the camera
        self.camera.start()
        
        # Initialize frame processor
        self.frame_processor = FrameProcessor()
        self.is_night_mode = False

    def get_frame(self):
        # Capture a frame as a NumPy array
        frame = self.camera.capture_array()
        # Convert from RGB (picamera2 default) to BGR (OpenCV default)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process frame based on conditions
        if self.is_night_mode:
            frame = self.frame_processor.enhance_night_vision(frame)
        return frame

    def toggle_night_mode(self):
        self.is_night_mode = not self.is_night_mode

    def release(self):
        # Stop and close the camera
        self.camera.stop()
