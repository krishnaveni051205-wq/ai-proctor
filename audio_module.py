import speech_recognition as sr
import threading
import time
import numpy as np
import librosa
class AudioModule:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300 
        self.recognizer.dynamic_energy_threshold = True
        self._is_speech_detected = False # Internal variable
        self.last_transcript=""
        self.alert_count = 0
        self.running = True
        self.current_volume=0
        self.is_live=True
        self.liveness_score=0.0
    def check_liveness(self,audio_data):
        # Convert raw bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        # 1. Calculate Spectral Centroid (the 'brightness' of the sound)
        centroid = librosa.feature.spectral_centroid(y=audio_np, sr=16000)[0]
        # 2. Calculate Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio_np, sr=16000)[0]
        avg_centroid = np.mean(centroid)
        # LOGIC: Digital speakers often lack the high-frequency 'air' of live speech.
        # If the centroid is too low or inconsistent, it's likely a recording.
        if avg_centroid < 1500: # Thresholds may need tuning for your mic
            self.is_live = False
            self.alert_count += 5
        else:
            self.is_live=True

    def _listen_in_background(self):
        """Monitors the microphone in a background thread."""
        with sr.Microphone() as source:
            # Calibrate for 1 second to ignore your room's fan/AC noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.running:
                try:
                    # Listen for a very short 0.5s burst
                    audio = self.recognizer.listen(source, timeout=0.5, phrase_time_limit=2.0)
                    self.current_volume=self.recognizer.energy_threshold/1000.0
                    raw_data=audio.get_raw_data(convert_rate=16000,convert_width=2)
                    self.check_liveness(raw_data)
                    try:
                        text = self.recognizer.recognize_google(audio)
                        self.last_transcript = text.lower()
                        self._is_speech_detected = True
                        print(f"Heard: {self.last_transcript}")
                        
                        # FIXED: Ensure all triggers are lowercase to match last_transcript
                        triggers = ["siri", "alexa", "google", "cortana", "chatgpt", "search", "browser",
                                    "formula", "definition", "example", "solve", "derive", "equation", "theorem",
                                    "tell me", "what is", "show me", "whisper", "speak up", "repeat",
                                    "matrix", "algorithm", "integral", "database", "complexity",
                                    "read it", "solve this", "what does it say", "check the book","probability","probability", "derivative", "limit", "summation", "coefficient"]
                        
                        # UPDATED: Loop through and count every trigger found
                        for word in triggers:
                            if word in self.last_transcript:
                                self.alert_count += 5
                                print(f"ALERT: Trigger found -> {word}. Score: {self.alert_count}")

                        time.sleep(0.8)
                        self._is_speech_detected = False
                        
                    except sr.UnknownValueError:
                        self.last_transcript = "[Unclear Speech]"
                        self._is_speech_detected = False
                except:
                    pass
    def start_stream(self):
        self.thread = threading.Thread(target=self._listen_in_background, daemon=True)
        self.thread.start()

    def is_speech(self):
        """This is the FUNCTION main.py will call."""
        return self._is_speech_detected

    def stop_stream(self):
        self.running = False
    def get_transcript(self):
        return self.last_transcript