import sounddevice as sd
import numpy as np

class AudioModule:
    def __init__(self, threshold=0.02):
        self.threshold = threshold
        self.current_volume = 0
        self.is_running = True

    def audio_callback(self, indata, frames, time, status):
        """This function is called for every block of audio captured."""
        if status:
            print(status)
        # Calculate RMS (Volume)
        self.current_volume = np.sqrt(np.mean(indata**2))

    def start_stream(self):
        # We start a non-blocking stream
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()

    def check_noise(self):
        # Returns True if volume exceeds our 'cheating' threshold
        return self.current_volume > self.threshold

    def stop_stream(self):
        self.stream.stop()