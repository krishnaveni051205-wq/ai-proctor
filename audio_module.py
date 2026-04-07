import speech_recognition as sr
import threading
import time

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
                    try:
                        text=self.recognizer.recognize_google(audio)
                        self.last_transcript=text.lower()
                        self._is_speech_detected = True
                        print(f"Heard: {self.last_transcript}")
                        triggers=["siri", "alexa", "google", "cortana", "chatgpt", "search", "browser",
                            "formula", "definition", "example", "solve", "derive", "equation", "theorem",
                            "tell me", "what is", "show me", "whisper", "speak up", "repeat",
                            "matrix", "algorithm", "integral", "database", "complexity"]
                        if any(word.lower() in self.last_transcript for word in triggers):
                            self.alert_count+=5
                        time.sleep(0.8)
                        self._is_speech_detected=False
                    except sr.UnknownValueError:
                        self.last_transcript="[Unlcear Speech]"
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    self._is_speech_detected = False

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