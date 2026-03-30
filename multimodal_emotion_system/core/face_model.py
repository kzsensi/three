import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from core.config import Config

class FaceEmotionModel:
    def __init__(self):
        # Initialize face detection (MediaPipe BlazeFace)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Initialize ONNX inference session for MobileNetV3 (Placeholder initialization)
        try:
            self.session = ort.InferenceSession(Config.FACE_MODEL_PATH, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            print(f"Warning: Failed to load ONNX model {Config.FACE_MODEL_PATH}. Using mock prediction.")
            self.session = None

    def preprocess(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run Face Detection
        results = self.face_detection.process(rgb_frame)
        if not results.detections:
            return None, 0.0 # No face detected, confidence 0
            
        # Extract highest confidence face
        detection = max(results.detections, key=lambda det: det.score[0])
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                     int(bboxC.width * iw), int(bboxC.height * ih)
                     
        # Ensure bounds
        x, y = max(0, x), max(0, y)
        face_crop = frame[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            return None, 0.0
            
        # Resize to MobileNetV3 input size (224x224)
        face_resized = cv2.resize(face_crop, (224, 224))
        # Normalize
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_tensor = np.expand_dims(face_normalized, axis=0)
        # HWC to CHW
        face_tensor = np.transpose(face_tensor, (0, 3, 1, 2))
        
        return face_tensor, detection.score[0]

    def predict(self, frame):
        """
        Predicts emotion from a raw BGR frame.
        Returns: logits/probabilities (shape: [7]), confidence score (0-1)
        """
        face_tensor, detect_score = self.preprocess(frame)
        
        if face_tensor is None:
            # Return uniform distribution if no face detected, with 0 confidence
            return np.ones(Config.NUM_CLASSES) / Config.NUM_CLASSES, 0.0
            
        if self.session:
            logits = self.session.run(None, {self.input_name: face_tensor})[0][0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            # Confidence heuristics: highest class prob scaled by face detection score
            confidence = float(np.max(probs) * detect_score)
            return probs, confidence
        else:
            # Mock Predict
            probs = np.random.dirichlet(np.ones(Config.NUM_CLASSES), size=1)[0]
            return probs, 0.8 * detect_score
