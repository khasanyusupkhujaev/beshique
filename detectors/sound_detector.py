import pyaudio
import numpy as np
from .base_detector import BaseDetector
import time
import logging

class SoundDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("BabyMonitor")
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.sound_threshold = 500   # General loudness
        self.cry_amplitude = 600     # Minimum amplitude for crying
        self.cry_freq_min = 250      # Hz, lower bound for cry frequency
        self.cry_freq_max = 1000     # Hz, upper bound for cry frequency
        self.cry_duration = 1.0      # Seconds of cry-like sound
        self.window_duration = 5.0
        self.sound_history = []
        self.cry_start_time = None
        self.audio = None
        self.stream = None
        self.is_active = False  # Default to inactive until proven otherwise

        try:
            self.logger.info("Initializing SoundDetector...")
            self.audio = pyaudio.PyAudio()
            device_count = self.audio.get_device_count()
            input_device_found = False
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    try:
                        self.stream = self.audio.open(
                            format=self.format,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            frames_per_buffer=self.chunk_size,
                            input_device_index=i
                        )
                        input_device_found = True
                        self.is_active = True
                        self.logger.info(f"SoundDetector: Using input device: {device_info['name']}")
                        break
                    except Exception as e:
                        self.logger.warning(f"SoundDetector: Failed to open device {device_info['name']}: {str(e)}")
                        continue

            if not input_device_found:
                self.logger.warning("SoundDetector: No valid audio input device found. Sound detection disabled.")
        except Exception as e:
            self.logger.error(f"SoundDetector: Error initializing PyAudio: {str(e)}. Sound detection disabled.")
            self.is_active = False
        finally:
            if not self.is_active and self.audio:
                self.audio.terminate()
                self.audio = None

    def detect_sound(self):
        """Detect sound amplitude and frequency"""
        if not self.is_active or not self.stream:
            return 0.0, 0.0
        
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            amplitude = np.sqrt(np.mean(audio_data**2))
            fft_data = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft_data)) * self.rate
            fft_magnitude = np.abs(fft_data)
            mask = (freqs >= self.cry_freq_min) & (freqs <= self.cry_freq_max)
            dominant_freq = freqs[mask][np.argmax(fft_magnitude[mask])] if np.any(mask) else 0.0
            return amplitude, dominant_freq
        except Exception as e:
            self.logger.error(f"SoundDetector: Error reading audio stream: {str(e)}")
            return 0.0, 0.0

    def process(self):
        """Process sound and detect crying with frequency"""
        if not self.is_active or not self.stream:
            return None, "Sound detection inactive or no audio device"

        try:
            current_time = time.time()
            amplitude, dominant_freq = self.detect_sound()
            
            self.sound_history.append((current_time, amplitude))
            while self.sound_history and current_time - self.sound_history[0][0] > self.window_duration:
                self.sound_history.pop(0)
            
            average_sound = sum(level for _, level in self.sound_history) / len(self.sound_history) if self.sound_history else 0.0

            is_loud = amplitude > self.sound_threshold
            is_cry_like = (amplitude > self.cry_amplitude) and \
                          (self.cry_freq_min <= dominant_freq <= self.cry_freq_max)
            is_crying = False
            
            if is_cry_like:
                if self.cry_start_time is None:
                    self.cry_start_time = current_time
                elif current_time - self.cry_start_time >= self.cry_duration:
                    is_crying = True
            else:
                self.cry_start_time = None

            sound_data = {
                'current_amplitude': amplitude,
                'average_sound': average_sound,
                'dominant_freq': dominant_freq,
                'is_loud': is_loud,
                'is_crying': is_crying
            }
            status_text = f"Sound: {amplitude:.0f}, Avg: {average_sound:.0f}, Freq: {dominant_freq:.0f}, Crying: {is_crying}"
            return sound_data, status_text
        except Exception as e:
            self.logger.error(f"SoundDetector: Error processing sound: {str(e)}")
            return None, "Sound detection error"

    def reset(self):
        """Reset sound history"""
        try:
            self.sound_history.clear()
            self.cry_start_time = None
        except Exception as e:
            self.logger.error(f"SoundDetector: Error resetting: {str(e)}")

    def cleanup(self):
        """Close audio stream"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
            self.logger.info("SoundDetector: Cleanup completed")
        except Exception as e:
            self.logger.error(f"SoundDetector: Error during cleanup: {str(e)}")