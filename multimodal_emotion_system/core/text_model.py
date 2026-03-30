import torch
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from core.config import Config

class TextEmotionModel:
    def __init__(self):
        # 1. Initialize Automatic Speech Recognition (Whisper-tiny for real-time edge use)
        try:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            self.asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        except Exception:
            print("Warning: Whisper weights not found locally. ASR will operate in Mock Mode.")
            self.asr_model = None

        # 2. Initialize Text Semantic Emotion classification
        try:
            # Using HuggingFace pipeline for the DistilRoBERTa-base classifier
            self.nlp_classifier = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base", 
                top_k=None # Returns all classes
            )
            # Map Hartmann's 7 classes (anger, disgust, fear, joy, neutral, sadness, surprise)
            # to our standard Config.CLASSES to keep order consistent.
            self.class_mapping = {
                'joy': 'Happy', 'sadness': 'Sad', 'anger': 'Angry',
                'surprise': 'Surprise', 'fear': 'Fear', 'disgust': 'Disgust', 'neutral': 'Neutral'
            }
        except Exception:
            print("Warning: NLP weights not found dynamically.")
            self.nlp_classifier = None

    def transcribe(self, audio_float32_array, sr=Config.SAMPLE_RATE):
        if self.asr_model is None:
            return "Mock transcription of the user speaking."
            
        input_features = self.processor(
            audio_float32_array, sampling_rate=sr, return_tensors="pt"
        ).input_features
        
        with torch.no_grad():
            predicted_ids = self.asr_model.generate(input_features)
            
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    def predict(self, text):
        """
        Takes raw string text, infers sentiment
        Returns: logits/probs [7], confidence
        """
        if not text or len(text) < 2:
            return np.ones(Config.NUM_CLASSES) / Config.NUM_CLASSES, 0.0
            
        if self.nlp_classifier:
            results = self.nlp_classifier(text)[0]
            
            # Results is list of dicts: [{'label': 'joy', 'score': 0.9}, ...]
            # We align it with Config.CLASSES order
            probs = np.zeros(Config.NUM_CLASSES)
            for res in results:
                mapped_label = self.class_mapping.get(res['label'], 'Neutral')
                idx = Config.CLASSES.index(mapped_label)
                probs[idx] = res['score']
                
            confidence = float(np.max(probs))
            return probs, confidence
        else:
            probs = np.random.dirichlet(np.ones(Config.NUM_CLASSES), size=1)[0]
            return probs, 0.7
