import cv2
import numpy as np
import mediapipe as mp
from .base_detector import BaseDetector
import time

class EyeDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.calibration_frames = []
        self.calibration_duration = 60  # Increased calibration duration
        self.open_eye_threshold = None
        self.closed_eye_threshold = None
        self.face_absent_timer = 0
        self.occlusion_threshold = 5.0
        self.brightness_history = []
        self.brightness_window = 10
        self.ear_smoothing_window = 5
        self.ear_history = []

    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR) with NumPy arrays"""
        eye_landmarks = np.array(eye_landmarks)
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def check_lighting(self, frame):
        """Check and enhance frame lighting dynamically"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        self.brightness_history.append(brightness)
        if len(self.brightness_history) > self.brightness_window:
            self.brightness_history.pop(0)

        avg_brightness = np.mean(self.brightness_history) if self.brightness_history else brightness

        clip_limit = max(1.0, 4.0 - (avg_brightness / 50.0))
        if avg_brightness < 40:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return frame

    def detect_eye_state(self, frame):
        """Detect eye state with occlusion handling and smoothed EAR"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        current_time = time.time()

        if results.multi_face_landmarks:
            self.face_absent_timer = current_time

            largest_face = max(results.multi_face_landmarks, key=lambda x:
                                (max([p.x for p in x.landmark]) - min([p.x for p in x.landmark])) *
                                (max([p.y for p in x.landmark]) - min([p.y for p in x.landmark])))
            landmarks = [(p.x * frame.shape[1], p.y * frame.shape[0]) for p in largest_face.landmark]

            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]

            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            self.ear_history.append(avg_ear)
            if len(self.ear_history) > self.ear_smoothing_window:
                self.ear_history.pop(0)
            smoothed_ear = np.mean(self.ear_history) if self.ear_history else avg_ear

            if len(self.calibration_frames) < self.calibration_duration:
                self.calibration_frames.append(smoothed_ear)
                return None, landmarks  # Return landmarks for potential ROI update

            if self.open_eye_threshold is None or self.closed_eye_threshold is None:
                if self.calibration_frames:
                    self.open_eye_threshold = max(self.calibration_frames) * 0.85
                    self.closed_eye_threshold = min(self.calibration_frames) * 1.15
                else:
                    return None, landmarks # Still calibrating or no face detected yet

            if self.open_eye_threshold is not None and self.closed_eye_threshold is not None:
                if smoothed_ear > (self.open_eye_threshold + self.closed_eye_threshold) / 2:
                    return True, landmarks
                else:
                    return False, landmarks
            else:
                return None, landmarks # Calibration not complete

        else:
            if self.face_absent_timer == 0:
                self.face_absent_timer = current_time
            elif current_time - self.face_absent_timer > self.occlusion_threshold:
                return "Occluded", None
            return None, None

    def process(self, frame):
        """Process frame and return eye state, modified frame, and status text"""
        if not self.is_active:
            return None, frame, "Eye detection inactive"

        # Enhance lighting if needed
        enhanced_frame = self.check_lighting(frame)
        eye_state, landmarks = self.detect_eye_state(enhanced_frame)

        status_text = ""
        # Overlay information on the frame
        if landmarks:
            # Define indices for eyes and mouth
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            mouth_indices = [61, 291, 0, 17]  # Outer lip landmarks

            # Draw only eye and mouth landmarks
            for idx in left_eye_indices + right_eye_indices + mouth_indices:
                point = landmarks[idx]
                cv2.circle(enhanced_frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

            left_eye = [landmarks[i] for i in left_eye_indices]
            ear_val = self.calculate_ear(left_eye)
            status_text = f"EAR: {ear_val:.2f}"
            if self.open_eye_threshold is not None and self.closed_eye_threshold is not None:
                status_text += f" (Open>{(self.open_eye_threshold + self.closed_eye_threshold) / 2:.2f})"
            if eye_state is True:
                status_text += " - Open"
            elif eye_state is False:
                status_text += " - Closed"
            elif eye_state == "Occluded":
                status_text = "Possible occlusion"

        else:
            status_text = "No face detected" if eye_state != "Occluded" else "Possible occlusion"

        cv2.putText(enhanced_frame, status_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Return eye state, processed frame, and status text
        return (eye_state, enhanced_frame, status_text)

    def reset(self):
        """Reset calibration and timers"""
        self.calibration_frames.clear()
        self.open_eye_threshold = None
        self.closed_eye_threshold = None
        self.face_absent_timer = 0
        self.brightness_history.clear()
        self.ear_history.clear()