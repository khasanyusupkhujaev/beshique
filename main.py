import cv2
import os
import queue
import signal
import subprocess
import sys
import time
from threading import Thread
from typing import Optional, Tuple, Dict, Any

from flask import Flask, Response, request, render_template_string, send_file
from flask_httpauth import HTTPBasicAuth

from camera.camera_manager import CameraManager
from detectors.eyes_detector import EyeDetector
from detectors.motion_detector import MotionDetector
from detectors.sound_detector import SoundDetector
from utils.config import Config  
from utils.logger import Logger

app = Flask(__name__)
auth = HTTPBasicAuth()

USERS = {"admin": "cradle123"}

@auth.verify_password
def verify_password(username, password):
    return USERS.get(username) == password

class BabyMonitor:
    def __init__(self):
        self.logger = Logger()
        self.logger.info("=== Initialising BabyMonitor ===")

        self.config = Config()  
        self.fps: float = getattr(self.config, "fps", 10)
        self.jpeg_quality: int = getattr(self.config, "jpeg_quality", 50)

        self.camera = CameraManager(resolution=self.config.resolution)
        self.eyes_detector = EyeDetector()
        self.motion_detector = MotionDetector()
        self.sound_detector = SoundDetector()

        self.frame_q: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=1)
        self._running = True
        Thread(target=self._capture_loop, daemon=True).start()

        self.current_state = "Unknown"
        self.previous_state: Optional[str] = None
        self.state_start_time: float = time.time()
        self.last_state_update: float = time.time()
        self.state_times: Dict[str, float] = {"Sleeping": 0.0, "Active": 0.0, "Crying": 0.0}
        self.bouncing_level: int = 0
        self.eye_state_history: list[Optional[bool]] = []

        self.logger.info("BabyMonitor ready (FPS=%s, JPEG=%s)" % (self.fps, self.jpeg_quality))

    def _capture_loop(self):
        """Continuously grab frames; replace queue item if full (drop old)."""
        while self._running:
            frame = self.camera.get_frame()
            if frame is None:
                self.logger.warning("Camera returned None frame")
                continue
            try:
                self.frame_q.put_nowait(frame)
            except queue.Full:
                _ = self.frame_q.get_nowait()
                self.frame_q.put_nowait(frame)

    def _classify_state(self, eye_state, motion_state, sound_state) -> str:
        """Return high‑level baby state (Sleeping / Active / Crying …)."""
        try:
            if eye_state is None or motion_state is None or sound_state is None:
                return "Unknown"

            eyes_closed = not eye_state[0] if eye_state[0] not in [None, "Occluded"] else None
            motion_score = motion_state[0]["motion_score"] if motion_state[0] else 0
            subtle_motion = motion_state[0]["subtle_motion"] if motion_state[0] else 0
            is_loud = sound_state[0]["is_loud"] if sound_state[0] else False
            is_crying = sound_state[0]["is_crying"] if sound_state[0] else False

            if eyes_closed is not None:
                if is_crying:
                    return "Crying"
                if motion_score > 25 or (subtle_motion > 0 and motion_score > 10) or is_loud:
                    return "Active"
                if eyes_closed and motion_score < 5 and not is_loud:
                    return "Sleeping"
                if not eyes_closed and motion_score < 25:
                    return "Awake/Calm"
            elif eye_state[0] == "Occluded":
                return "Occluded"
            return "Unknown"
        except Exception as exc:
            self.logger.error(f"State classification error: {exc}")
            return "Unknown"

    def _update_bouncing_level(self, current_state, motion_data, sound_data, eyes_open):
        """Update bouncing level and state times."""
        try:
            current_time = time.time()
            motion_score = motion_data['motion_score'] if motion_data else 0
            average_motion = motion_data['average_motion'] if motion_data else 0
            subtle_motion = motion_data['subtle_motion'] if motion_data else 0
            motion_trend = motion_data['motion_trend'] if motion_data else 0
            sound_amplitude = sound_data['current_amplitude'] if sound_data else 0
            average_sound = sound_data['average_sound'] if sound_data else 0
            is_loud = sound_data['is_loud'] if sound_data else False
            is_crying = sound_data['is_crying'] if sound_data else False
            
            if eyes_open is not None:
                self.eye_state_history.append(eyes_open)
                if len(self.eye_state_history) > 5:  
                    self.eye_state_history.pop(0)
            
            # Update state times
            if self.current_state not in ["Unknown", "Occluded"]:
                duration = current_time - self.last_state_update
                if is_crying:
                    self.state_times["Crying"] += duration
                elif self.current_state == "Sleeping" and all(not state for state in self.eye_state_history[-5:] if state is not None):
                    self.state_times["Sleeping"] += duration
                elif self.current_state in ["Active", "Awake/Calm"]:
                    self.state_times["Active"] += duration
            self.last_state_update = current_time

            # Update state transition tracking
            if current_state != self.previous_state:
                self.state_start_time = current_time
                self.previous_state = current_state
            
            state_duration = current_time - self.state_start_time
            
            # Update bouncing level
            if state_duration >= 0.5:
                if eyes_open == "Occluded":
                    self.logger.info(f"Bouncing level unchanged due to occlusion: {self.bouncing_level}")
                    return self.bouncing_level
                
                if is_crying:
                    self.bouncing_level = 3
                elif (current_state == "Active" and average_motion > 0.8) or \
                     (motion_trend > 0.5 and subtle_motion > 0) or \
                     (is_loud and average_sound > 800):
                    self.bouncing_level = 2
                elif eyes_open or (average_motion > 0.5 and motion_score > 10) or \
                     (average_sound > 500 and not is_crying):
                    self.bouncing_level = 1
                elif current_state == "Sleeping" or (motion_score < 5 and not is_loud):
                    self.bouncing_level = 0
            
            self.logger.info(f"Bouncing level updated: {self.bouncing_level}")
            return self.bouncing_level
        except Exception as e:
            self.logger.error(f"Error in _update_bouncing_level: {str(e)}")
            return self.bouncing_level

    def generate_frames(self):
        """Yield MJPEG stream; will drop frames if client is too slow."""
        frame_interval = 1 / self.fps
        while True:
            try:
                start_time = time.time()

                # Pop the latest frame; if none available within 0.2 s keep looping
                try:
                    frame = self.frame_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                # Process detectors -----------------------------------
                eye_result = self.eyes_detector.process(frame)
                eyes_open = eye_result[0] if eye_result else None
                processed_frame = eye_result[1] if eye_result else frame
                landmarks = eye_result[1] if eye_result and eyes_open in [True, False] else None

                motion_state = self.motion_detector.process(processed_frame, face_landmarks=landmarks)
                sound_state = self.sound_detector.process()

                self.current_state = self._classify_state(eye_result, motion_state, sound_state)

                # Update bouncing level and state times
                motion_data = motion_state[0] if motion_state and motion_state[0] else None
                sound_data = sound_state[0] if sound_state and sound_state[0] else None
                self.bouncing_level = self._update_bouncing_level(self.current_state, motion_data, sound_data, eyes_open)
                # Overlay --------------------------------------------
                cv2.putText(processed_frame, f"State: {self.current_state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(processed_frame, f"Bouncing level: {self.bouncing_level}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # Resize + encode ------------------------------------
                stream_frame = cv2.resize(processed_frame, (640, 480))
                ret, jpeg = cv2.imencode('.jpg', stream_frame,
                                         [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if not ret:
                    self.logger.error("JPEG encode failed")
                    continue

                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

                # Frame pacing ---------------------------------------
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
            except GeneratorExit:
                # Client disconnected
                break
            except Exception as exc:
                self.logger.error(f"MJPEG loop error: {exc}")
                time.sleep(1)

    def get_statistics(self):
        return {k: round(v, 1) for k, v in self.state_times.items()}

    def cleanup(self):
        self._running = False
        try:
            self.camera.release()
            self.sound_detector.cleanup()
            self.logger.info("Cleanup done, exiting.")
        except Exception as exc:
            self.logger.error(f"Cleanup error: {exc}")

def check_ip():
    """Check and verify the IP address of wlan0"""
    try:
        ip_result = subprocess.run(['ip', 'addr', 'show', 'wlan0'], 
                                  capture_output=True, text=True)
        monitor.logger.info(f"Current wlan0 status: {ip_result.stdout}")
        
        if "192.168.4.1" in ip_result.stdout:
            return "192.168.4.1"
        
        import re
        ip_match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', ip_result.stdout)
        if ip_match:
            return ip_match.group(1)
        return None
    except Exception as e:
        monitor.logger.error(f"Error checking IP: {str(e)}")
        return None

def is_wifi_connected():
    """Check if the RPi is connected to a Wi-Fi network and ensure interface is up."""
    try:
        up_result = subprocess.run(['ip', 'link', 'show', 'wlan0'], 
                                capture_output=True, text=True, check=False)
        if "state DOWN" in up_result.stdout:
            subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'up'], check=False)
            time.sleep(1)
            
        result = subprocess.run(['nmcli', '-t', '-f', 'STATE,DEVICE', 'connection', 'show', '--active'],
                               capture_output=True, text=True, check=False)
        for line in result.stdout.splitlines():
            if ':' in line:
                state, device = line.split(':')
                if state == 'activated' and 'wlan' in device:
                    return True
        return False
    except Exception as e:
        monitor.logger.error(f"Error checking Wi-Fi status: {str(e)}")
        return False

def setup_hotspot():
    """Set up Wi-Fi hotspot with sudo privileges."""
    try:
        monitor.logger.info("Setting up Wi-Fi hotspot...")
        subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'down'], check=False)
        time.sleep(1)
        subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'up'], check=False)
        time.sleep(1)
        subprocess.run(['sudo', 'systemctl', 'stop', 'hostapd'], check=False)
        subprocess.run(['sudo', 'systemctl', 'stop', 'dnsmasq'], check=False)
        subprocess.run(['sudo', 'nmcli', 'dev', 'disconnect', 'wlan0'], check=False)
        subprocess.run(['sudo', 'ip', 'addr', 'flush', 'dev', 'wlan0'], check=False)
        subprocess.run(['sudo', 'ip', 'addr', 'add', '192.168.4.1/24', 'dev', 'wlan0'], check=False)
        time.sleep(1)
        current_ip = check_ip()
        if current_ip != "192.168.4.1":
            monitor.logger.error(f"Failed to set static IP. Current IP: {current_ip}")
            return False
        subprocess.run(['sudo', 'systemctl', 'restart', 'dnsmasq'], check=False)
        time.sleep(1)
        subprocess.run(['sudo', 'systemctl', 'restart', 'hostapd'], check=False)
        time.sleep(2)
        hostapd_status = subprocess.run(['systemctl', 'is-active', 'hostapd'], 
                                       capture_output=True, text=True).stdout.strip()
        dnsmasq_status = subprocess.run(['systemctl', 'is-active', 'dnsmasq'], 
                                       capture_output=True, text=True).stdout.strip()
        monitor.logger.info(f"Hotspot services: hostapd={hostapd_status}, dnsmasq={dnsmasq_status}")
        monitor.logger.info("Hotspot activated: SSID=beshique-setup, Password=cradle123")
        return True
    except Exception as e:
        monitor.logger.error(f"Failed to set up hotspot: {str(e)}")
        return False

def try_connect_wifi(ssid, password):
    """Attempt to connect to Wi-Fi with provided credentials."""
    try:
        result = subprocess.run(['sudo', 'nmcli', 'dev', 'wifi', 'connect', ssid, 'password', password],
                               capture_output=True, text=True, check=False)
        if result.returncode == 0:
            monitor.logger.info(f"Connected to Wi-Fi: {ssid}")
            return True
        else:
            monitor.logger.error(f"Wi-Fi connection failed: {result.stderr}")
            return False
    except Exception as e:
        monitor.logger.error(f"Wi-Fi connection failed: {str(e)}")
        return False

def monitor_wifi():
    """Monitor Wi-Fi connection and manage hotspot."""
    while True:
        try:
            subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'up'], check=False)
            time.sleep(1)
            wifi_connected = is_wifi_connected()
            hostapd_running = os.path.exists('/run/hostapd.pid')
            current_ip = check_ip()
            monitor.logger.info(f"WiFi status: connected={wifi_connected}, hostapd={hostapd_running}, IP={current_ip}")
            if "state DOWN" in subprocess.run(['ip', 'link', 'show', 'wlan0'], 
                                         capture_output=True, text=True).stdout:
                monitor.logger.warning("wlan0 still down after setting up - trying network restart")
                subprocess.run(['sudo', 'systemctl', 'restart', 'NetworkManager'], check=False)
                time.sleep(5)
            if not wifi_connected and not hostapd_running:
                monitor.logger.info("WiFi disconnected and no hotspot active, starting hotspot...")
                if setup_hotspot():
                    monitor.logger.info("Restarting Flask to bind to hotspot IP")
                    os.execv(sys.executable, ['python'] + sys.argv)
            elif wifi_connected and hostapd_running:
                monitor.logger.info("WiFi connected but hotspot still active - stopping hotspot")
                subprocess.run(['sudo', 'systemctl', 'stop', 'hostapd'], check=False)
                subprocess.run(['sudo', 'systemctl', 'stop', 'dnsmasq'], check=False)
            time.sleep(30)
        except Exception as e:
            monitor.logger.error(f"Error in WiFi monitor thread: {str(e)}")
            time.sleep(60)

monitor = BabyMonitor()

def _graceful_shutdown(signum, frame):
    monitor.cleanup()
    sys.exit(0)

for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, _graceful_shutdown)

@app.route('/')
@auth.login_required
def index():
    return render_template_string(
        """
        <html><head><title>Beshique Monitor</title></head>
        <body style='text-align:center;font-family:Arial'>
            <h2>Beshique Baby Monitor</h2>
            <img src="{{ url_for('video_feed') }}" width="640" height="480"/>
            <p><a href="{{ url_for('statistics') }}">Statistics</a> |
               <a href="{{ url_for('snapshot') }}">Snapshot</a> |
               <a href="{{ url_for('setup') }}">Wi-Fi Setup</a></p>
        </body></html>
        """)

@app.route('/video_feed')
@auth.login_required
def video_feed():
    return Response(monitor.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot.jpg')
@auth.login_required
def snapshot():
    try:
        frame = monitor.frame_q.get(timeout=0.5)
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
    except queue.Empty:
        return "No frame", 503

@app.route('/statistics')
@auth.login_required
def statistics():
    stats = monitor.get_statistics()
    tpl = """
    <html><head><title>Stats</title></head>
    <body style='text-align:center;font-family:Arial'>
        <h2>State statistics (seconds)</h2>
        {% for k, v in stats.items() %}<p>{{ k }}: {{ v }}</p>{% endfor %}
        <p><a href="{{ url_for('index') }}">Back</a></p>
    </body></html>"""
    return render_template_string(tpl, stats=stats)

@app.route('/setup', methods=['GET', 'POST'])
@auth.login_required
def setup():
    """Wi-Fi setup page"""
    if request.method == 'POST':
        ssid = request.form['ssid']
        password = request.form['password']
        try:
            if try_connect_wifi(ssid, password):
                wifi_config_path = '/home/admin/baby_crib/utils/wifi_config.txt'
                os.makedirs(os.path.dirname(wifi_config_path), exist_ok=True)
                with open(wifi_config_path, 'w') as f:
                    f.write(f"{ssid}\n{password}")
                ip_result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
                ip_address = ip_result.stdout.strip().split()[0] if ip_result.stdout.strip() else "unknown"
                response = (f"Wi-Fi configured! Reconnect to your network and visit http://{ip_address}:5000. "
                           "Hotspot stopping in 5 seconds...")
                def stop_hotspot():
                    time.sleep(5)
                    try:
                        subprocess.run(['sudo', 'systemctl', 'stop', 'hostapd'], check=False)
                        subprocess.run(['sudo', 'systemctl', 'stop', 'dnsmasq'], check=False)
                        subprocess.run(['sudo', 'nmcli', 'dev', 'disconnect', 'wlan0'], capture_output=True)
                        monitor.logger.info("Hotspot stopped successfully")
                    except Exception as e:
                        monitor.logger.error(f"Error stopping hotspot: {str(e)}")
                Thread(target=stop_hotspot).start()
                return response
            else:
                return "Error: Failed to connect to Wi-Fi. Please check credentials and try again."
        except Exception as e:
            monitor.logger.error(f"Setup error: {str(e)}")
            return f"Error: {str(e)}"
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wi-Fi Setup</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f4; text-align: center; padding-top: 50px; }
            form { display: inline-block; text-align: left; }
            input { margin: 10px 0; padding: 5px; width: 200px; }
            input[type="submit"] { background-color: #4CAF50; color: white; border: none; padding: 10px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Setup Wi-Fi</h1>
        <form method="post">
            <label>Wi-Fi Name (SSID):</label><br>
            <input type="text" name="ssid" required><br>
            <label>Password:</label><br>
            <input type="password" name="password" required><br>
            <input type="submit" value="Connect">
        </form>
    </body>
    </html>
    """)

wifi_config_path = '/home/admin/baby_crib/utils/wifi_config.txt'
os.makedirs(os.path.dirname(wifi_config_path), exist_ok=True)

if is_wifi_connected():
    monitor.logger.info("WiFi already connected, proceeding to Flask server.")
else:
    monitor.logger.info("No WiFi connection detected.")
    subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'up'], check=False)
    if os.path.exists(wifi_config_path):
        monitor.logger.info("Found WiFi config, attempting to connect...")
        try:
            with open(wifi_config_path, 'r') as f:
                content = f.read().strip()
                if '\n' in content:
                    ssid, password = content.split('\n')
                    if try_connect_wifi(ssid, password):
                        monitor.logger.info("WiFi connected successfully.")
                    else:
                        monitor.logger.info("WiFi connection failed, activating hotspot...")
                        setup_hotspot()
                else:
                    monitor.logger.error("Invalid WiFi config format")
                    setup_hotspot()
        except Exception as e:
            monitor.logger.error(f"Error reading/using WiFi config: {str(e)}")
            setup_hotspot()
    else:
        monitor.logger.info("No WiFi config found, activating hotspot...")
        setup_hotspot()

if __name__ == "__main__":
    try:
        Thread(target=monitor_wifi, daemon=True).start()
        current_ip = check_ip()
        monitor.logger.info(f"Starting Flask server with detected IP: {current_ip}")
        if os.path.exists('/run/hostapd.pid') and current_ip == "192.168.4.1":
            monitor.logger.info("Starting Flask server in hotspot mode")
            app.run(host='192.168.4.1', port=5000, threaded=True)
        else:
            monitor.logger.info("Starting Flask server in normal mode")
            app.run(host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        monitor.logger.error(f"Server failed: {e}")
    finally:
        monitor.cleanup()
