import cv2
import numpy as np
from .base_detector import BaseDetector
import time

class MotionDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        self.motion_threshold = 15
        self.window_duration = 5.0
        self.motion_history = []
        self.previous_frame = None
        self.roi = None
        self.roi_selected = False
        self.current_score = 0.0
        self.average_motion = 0.0
        self.subtle_motion = 0.0
        self.motion_trend = 0.0

    def set_roi(self, frame_shape, face_landmarks=None):
        """Set ROI dynamically based on face position if available"""
        height, width = frame_shape[:2]
        if face_landmarks and not self.roi_selected:
            x_coords = [p[0] for p in face_landmarks]
            y_coords = [p[1] for p in face_landmarks]
            face_x, face_y = int(np.mean(x_coords)), int(np.mean(y_coords))
            w, h = int(width * 0.4), int(height * 0.4)
            x = max(0, face_x - w // 2)
            y = max(0, face_y - h // 2)
            w = min(w, width - x)
            h = min(h, height - y)
            self.roi = (x, y, w, h)
        elif not self.roi_selected:
            x, y = int(width * 0.2), int(height * 0.2)
            w, h = int(width * 0.6), int(height * 0.6)
            self.roi = (x, y, w, h)
        self.roi_selected = True

    def detect_motion(self, frame):
        """Detect motion with subtle movement sensitivity"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return 0.0

        frame_diff = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        self.subtle_motion = motion_percentage if 2 < motion_percentage < 10 else 0
        self.previous_frame = gray
        return motion_percentage

    def process(self, frame, face_landmarks=None):
        """Process frame with motion trend analysis and return modified frame"""
        if not self.is_active:
            return None, frame, "Motion detection inactive"

        if not self.roi_selected:
            self.set_roi(frame.shape, face_landmarks)

        x, y, w, h = self.roi
        roi_frame = frame[y:y+h, x:x+w]

        current_time = time.time()
        motion_percentage = self.detect_motion(roi_frame)
        motion_score = min(motion_percentage / 10, 10.0)

        self.motion_history.append((current_time, motion_score))
        while self.motion_history and current_time - self.motion_history[0][0] > self.window_duration:
            self.motion_history.pop(0)

        if self.motion_history:
            self.average_motion = sum(score for _, score in self.motion_history) / len(self.motion_history)
            if len(self.motion_history) > 2:
                t0, s0 = self.motion_history[0]
                t1, s1 = self.motion_history[-1]
                self.motion_trend = (s1 - s0) / (t1 - t0) if t1 > t0 else 0
            else:
                self.motion_trend = 0
        else:
            self.average_motion = 0.0
            self.motion_trend = 0

        self.current_score = sum(score for _, score in self.motion_history)

        # Overlay ROI and motion stats on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        status_text = f"Score: {self.current_score:.1f}, Avg: {self.average_motion:.1f}, Trend: {self.motion_trend:.2f}"
        cv2.putText(frame, status_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        motion_data = {
            'motion_score': self.current_score,
            'average_motion': self.average_motion,
            'subtle_motion': self.subtle_motion,
            'motion_trend': self.motion_trend
        }
        return motion_data, frame, status_text

    def reset(self):
        """Reset detector state"""
        self.previous_frame = None
        self.motion_history.clear()
        self.current_score = 0.0
        self.average_motion = 0.0
        self.subtle_motion = 0.0
        self.motion_trend = 0.0
        self.roi_selected = False