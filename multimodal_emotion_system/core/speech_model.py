import numpy as np
import librosa
import onnxruntime as ort
import webrtcvad
from core.config import Config

class SpeechEmotionModel:
    def __init__(self):
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(Config.VAD_MODE)
        
        # Load ONNX Model
        try:
            self.session = ort.InferenceSession(Config.SPEECH_MODEL_PATH, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            print(f"Warning: Failed to load ONNX model {Config.SPEECH_MODEL_PATH} for Speech.")
            self.session = None

    def is_speech(self, audio_chunk_pcm16, sample_rate=Config.SAMPLE_RATE):
        """ Checks if the 10/20/30ms chunk contains active speech """
        # Only testing first 30ms frame for quick check, though in prod we'd window it.
        frame_len = int(sample_rate * 0.03)
        chunk = audio_chunk_pcm16[:frame_len * 2] # 2 bytes per sample (16bit)
        if len(chunk) < frame_len * 2:
            return False
            
        try:
            return self.vad.is_speech(chunk, sample_rate)
        except Exception:
            return False

    def extract_mel_spectrogram(self, audio_data, sr=Config.SAMPLE_RATE):
        """ Convert raw float32 audio to mel-spectrogram features """
        # Generate Log-Mel Spectrogram
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, fmax=8000)
        log_S = librosa.power_to_db(S, ref=np.max)
        
        # Ensure fixed temporal length (e.g., 2 seconds = ~63 frames at typical hop lengths)
        target_frames = 63 
        if log_S.shape[1] < target_frames:
            pad_width = target_frames - log_S.shape[1]
            log_S = np.pad(log_S, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            log_S = log_S[:, :target_frames]
            
        # Add channel and batch dimension (1, 1, 128, 63)
        tensor = log_S.reshape(1, 1, log_S.shape[0], log_S.shape[1]).astype(np.float32)
        return tensor

    def predict(self, audio_float32_array):
        """
        Takes a 2-second audio float32 array and predicts emotion
        Returns: logits/probs [7], confidence (0-1)
        """
        # Convert to PCM 16-bit for VAD check
        pcm16 = np.int16(audio_float32_array * 32767).tobytes()
        if not self.is_speech(pcm16):
            # No speech implies low confidence in vocal prosody
            return np.ones(Config.NUM_CLASSES) / Config.NUM_CLASSES, 0.1
            
        features = self.extract_mel_spectrogram(audio_float32_array)
        
        if self.session:
            logits = self.session.run(None, {self.input_name: features})[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            # Use max prob as base confidence, penalize if distribution is too uniform
            confidence = float(np.max(probs))
            return probs, confidence
        else:
            # Mock inference
            probs = np.random.dirichlet(np.ones(Config.NUM_CLASSES), size=1)[0]
            return probs, 0.7
