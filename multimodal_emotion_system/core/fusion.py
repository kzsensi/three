import torch
import torch.nn as nn
import numpy as np
from core.config import Config

class AdaptiveAttentionFusion(nn.Module):
    """
    To be trained dynamically, this PyTorch model learns
    how to weigh Face, Speech, and Text based on their output probability
    entropies and confidence heuristics.
    """
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(AdaptiveAttentionFusion, self).__init__()
        # Input size = 3 modalities * (7 classes + 1 confidence) = 24 features
        input_dim = 3 * (num_classes + 1)
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        
        # Attention layer to generate 3 weights for the modalities
        self.attention_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, face_feats, speech_feats, text_feats):
        """
        Features are standard Tensors: [batch_size, num_classes + 1]
        """
        # Form [batch_size, 24]
        combined = torch.cat((face_feats, speech_feats, text_feats), dim=-1)
        
        hidden = self.relu(self.fc1(combined))
        
        # Get attention weights [batch_size, 3] -> (w_face, w_speech, w_text)
        attn_weights = self.attention_net(hidden)
        w_f, w_s, w_t = attn_weights[:, 0:1], attn_weights[:, 1:2], attn_weights[:, 2:3]
        
        # We only apply attention to the probability vectors, ignoring the confidence scalar at index -1
        face_probs = face_feats[:, :-1]
        speech_probs = speech_feats[:, :-1]
        text_probs = text_feats[:, :-1]
        
        # Fuse probabilities
        fused_probs = (w_f * face_probs) + (w_s * speech_probs) + (w_t * text_probs)
        return fused_probs, attn_weights

class FusionEngine:
    def __init__(self):
        self.model = AdaptiveAttentionFusion()
        try:
            self.model.load_state_dict(torch.load(Config.FUSION_MODEL_PATH))
            self.model.eval()
        except:
            print("Warning: Adaptive Fusion weights not found, using Untrained Initialization.")
            
    def predict(self, face_state, speech_state, text_state):
        """
        Inputs evaluate to dictionaries: {'val': probs_array, 'conf': scalar}
        """
        # Handle cases where a stream has not initialized yet (None)
        dummy_probs = np.ones(Config.NUM_CLASSES) / Config.NUM_CLASSES
        
        f_val = face_state['val'] if face_state else dummy_probs
        f_conf = face_state['conf'] if face_state else 0.0
        
        s_val = speech_state['val'] if speech_state else dummy_probs
        s_conf = speech_state['conf'] if speech_state else 0.0
        
        t_val = text_state['val'] if text_state else dummy_probs
        t_conf = text_state['conf'] if text_state else 0.0
        
        # Prepare tensors
        f_feat = torch.tensor(np.append(f_val, f_conf), dtype=torch.float32).unsqueeze(0)
        s_feat = torch.tensor(np.append(s_val, s_conf), dtype=torch.float32).unsqueeze(0)
        t_feat = torch.tensor(np.append(t_val, t_conf), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            fused_probs, att_weights = self.model(f_feat, s_feat, t_feat)
        
        fused_probs_np = fused_probs.squeeze().numpy()
        att_np = att_weights.squeeze().numpy()
        
        final_class_idx = np.argmax(fused_probs_np)
        
        return {
            "emotion": Config.CLASSES[final_class_idx],
            "confidence": float(fused_probs_np[final_class_idx]),
            "probabilities": fused_probs_np.tolist(),
            "attention_weights": {
                "face": float(att_np[0]),
                "speech": float(att_np[1]),
                "text": float(att_np[2])
            }
        }
