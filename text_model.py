from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class MedicalTextModel:
    def __init__(self):
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
        self.chest_labels = ['Normal', 'Pneumonia', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration']
        self.brain_labels = ['Normal', 'Brain_Tumor', 'Stroke', 'Multiple_Sclerosis', 'Hemorrhage']
        self.chest_keywords = {
            'Pneumonia': ['cough', 'fever', 'chest pain', 'shortness of breath', 'fatigue'],
            'Cardiomegaly': ['chest pain', 'shortness of breath', 'fatigue', 'swelling', 'palpitations'],
            'Atelectasis': ['shortness of breath', 'cough', 'chest pain'],
            'Effusion': ['shortness of breath', 'chest pain', 'dry cough'],
            'Normal': ['healthy', 'fine', 'good', 'normal', 'no symptoms']
        }
        self.brain_keywords = {
            'Brain_Tumor': ['headache', 'seizure', 'vision problems', 'nausea', 'confusion'],
            'Stroke': ['sudden weakness', 'speech problems', 'face drooping', 'arm weakness'],
            'Multiple_Sclerosis': ['fatigue', 'numbness', 'vision problems', 'weakness'],
            'Hemorrhage': ['severe headache', 'nausea', 'vomiting', 'confusion'],
            'Normal': ['healthy', 'fine', 'good', 'normal', 'no symptoms']
        }
        
    def predict(self, symptoms_text):
        # Auto-detect based on symptoms
        brain_symptoms = ['headache', 'seizure', 'vision', 'confusion', 'weakness', 'speech']
        is_brain = any(symptom in symptoms_text.lower() for symptom in brain_symptoms)
        
        if is_brain:
            keywords = self.brain_keywords
            labels = self.brain_labels
        else:
            keywords = self.chest_keywords
            labels = self.chest_labels
            
        keyword_scores = self._calculate_keyword_scores(symptoms_text.lower(), keywords)
        
        if not keyword_scores or max(keyword_scores.values()) < 0.2:
            keyword_scores['Normal'] = 0.8
        
        total = sum(keyword_scores.values()) or 1
        probabilities = {k: v/total for k, v in keyword_scores.items()}
        
        for label in labels:
            if label not in probabilities:
                probabilities[label] = 0.05
        
        best_prediction = max(probabilities, key=probabilities.get)
        
        return {
            'prediction': best_prediction,
            'confidence': probabilities[best_prediction],
            'probabilities': probabilities
        }
    
    def _calculate_keyword_scores(self, text, keywords_dict):
        scores = {}
        for condition, keywords in keywords_dict.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > 0:
                scores[condition] = matches / len(keywords) + 0.3
        return scores
    
    def get_attention_weights(self, symptoms_text):
        inputs = self.tokenizer(symptoms_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attention = outputs.attentions[-1].mean(dim=1).squeeze()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
        return list(zip(tokens, attention[0].numpy()))