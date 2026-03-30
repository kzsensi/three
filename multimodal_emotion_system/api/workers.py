import asyncio
from typing import Dict, Any

class WorkerPool:
    def __init__(self, face_model, speech_model, text_model, fusion_engine):
        self.face_model = face_model
        self.speech_model = speech_model
        self.text_model = text_model
        self.fusion_engine = fusion_engine
        
        # Async queues for incoming data
        self.face_queue = asyncio.Queue(maxsize=30)
        self.audio_queue = asyncio.Queue(maxsize=10)
        self.result_queue = asyncio.Queue()
        
    async def face_worker(self):
        """ Processes video frames as fast as possible up to MAX_FPS config """
        while True:
            frame = await self.face_queue.get()
            try:
                # Typically takes <30ms on CPU with MobileNet / MediaPipe
                probs, conf = self.face_model.predict(frame)
                await self.result_queue.put({'modality': 'face', 'val': probs, 'conf': conf})
            except Exception as e:
                print(f"Face Error: {str(e)}")
            finally:
                self.face_queue.task_done()

    async def audio_worker(self):
        """ Handles 2-second overlapping audio chunks for speech+text """
        while True:
            chunk = await self.audio_queue.get()
            # 1. Check VAD
            if self.speech_model.is_speech(chunk):
                # Run Acoustic Feature Extractor (CNN-BiLSTM)
                audio_probs, audio_conf = self.speech_model.predict(chunk)
                await self.result_queue.put({'modality': 'speech', 'val': audio_probs, 'conf': audio_conf})
                
                # 2. Transcribe and Semantic Sentiment (Whisper + RoBERTa)
                # Note: To avoid blocking the event loop on huge models, 
                # run_in_executor should be used in production
                transcription = self.text_model.transcribe(chunk)
                if transcription:
                    text_probs, text_conf = self.text_model.predict(transcription)
                    await self.result_queue.put({'modality': 'text', 'val': text_probs, 'conf': text_conf})
            self.audio_queue.task_done()
            
    async def get_fused_result_stream(self):
        """ Yields the latest fused emotion whenever a modality updates """
        # Holds the most recently calculated state of each modality
        state: Dict[str, Any] = {'face': None, 'speech': None, 'text': None}
        
        while True:
            # Wait for any modality to produce a new result
            new_result = await self.result_queue.get()
            modality = new_result['modality']
            state[modality] = {'val': new_result['val'], 'conf': new_result['conf']}
            
            # Predict the joined late-fusion result
            fused_out = self.fusion_engine.predict(
                face_state=state['face'],
                speech_state=state['speech'],
                text_state=state['text']
            )
            yield fused_out
            self.result_queue.task_done()
