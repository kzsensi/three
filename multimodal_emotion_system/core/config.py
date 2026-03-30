class Config:
    # Emotion Classes definitions
    CLASSES = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust', 'Neutral']
    NUM_CLASSES = 7
    
    # Model Paths
    FACE_MODEL_PATH = "models/checkpoints/face_mobilenetv3.onnx"
    SPEECH_MODEL_PATH = "models/checkpoints/speech_cnn_bilstm.onnx"
    TEXT_MODEL_NAME = "distilroberta-base"
    FUSION_MODEL_PATH = "models/checkpoints/adaptive_fusion.pt"
    
    # Stream & Latency Configuration
    MAX_FPS_FACE = 10
    AUDIO_CHUNK_SIZE_SEC = 2.0
    SAMPLE_RATE = 16000
    VAD_MODE = 3 # 0-3 (Aggressiveness)
    
    # Confidence Thresholds
    CONFIDENCE_THRESHOLD = 0.65
