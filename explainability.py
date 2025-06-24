import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class ExplainabilityEngine:
    def __init__(self):
        pass
    
    def generate_gradcam_heatmap(self, attention_map, original_image):
        # Resize attention map to match image size
        heatmap = cv2.resize(attention_map, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert PIL to numpy if needed
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image.resize((224, 224)))
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        return overlay
    
    def generate_text_explanation(self, image_pred, text_pred, attention_tokens):
        explanation = f"""
        **Diagnosis Analysis:**
        
        **Image Analysis:** {image_pred['prediction']} (Confidence: {image_pred['confidence']:.2%})
        **Top Image Findings:** {', '.join([f"{pred[0]} ({pred[1]:.1%})" for pred in image_pred.get('top_predictions', [])])}
        **Symptom Analysis:** {text_pred['prediction']} (Confidence: {text_pred['confidence']:.2%})
        
        **Key Findings:**
        - Primary diagnosis based on imaging: {image_pred['prediction']}
        - Symptom correlation: {text_pred['prediction']}
        
        **Important Symptoms Detected:**
        {self._format_attention_tokens(attention_tokens[:5])}
        
        **Recommendation:** 
        {'Consult a healthcare professional for further evaluation.' if max(image_pred['confidence'], text_pred['confidence']) < 0.8 else 'High confidence prediction - recommend immediate medical attention.'}
        """
        return explanation
    
    def _format_attention_tokens(self, tokens):
        return "\n".join([f"- {token}: {weight:.3f}" for token, weight in tokens if not token.startswith('[') and len(token) > 2])
    
    def combine_predictions(self, image_pred, text_pred):
        # Weighted ensemble with confidence-based weighting
        image_weight = 0.7  # Images typically more reliable for medical diagnosis
        text_weight = 0.3
        
        combined_probs = {}
        for label in image_pred['probabilities']:
            img_prob = image_pred['probabilities'][label]
            txt_prob = text_pred['probabilities'].get(label, 0)
            combined_probs[label] = image_weight * img_prob + text_weight * txt_prob
        
        # Apply confidence threshold
        final_prediction = max(combined_probs, key=combined_probs.get)
        final_confidence = combined_probs[final_prediction]
        
        # If confidence too low, default to most likely from top predictions
        if final_confidence < 0.3:
            top_image = image_pred.get('top_predictions', [(image_pred['prediction'], image_pred['confidence'])])[0]
            final_prediction = top_image[0]
            final_confidence = top_image[1] * 0.8  # Reduce confidence for uncertainty
        
        return {
            'prediction': final_prediction,
            'confidence': min(final_confidence, 0.95),  # Cap confidence
            'probabilities': combined_probs
        }