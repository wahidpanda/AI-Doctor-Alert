import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import requests
from model_downloader import ModelDownloader

class MedicalBERTPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or "models/medical_bert_model"
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load model from downloaded path"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Medical BERT model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to rule-based system
            self.model = None
    
    def predict_urgency_with_confidence(self, text):
        """Predict medical urgency with confidence score"""
        # If model is not loaded, use rule-based fallback
        if self.model is None:
            return self._rule_based_prediction(text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, dim=-1)
            
            # Map to urgency levels
            urgency_levels = ["Low", "Medium", "High"]
            urgency = urgency_levels[predicted_class.item()]
            confidence_score = confidence.item()
            
            # Determine patient status and alarm
            patient_status, alarm_status = self._get_patient_status(urgency, confidence_score)
            
            return urgency, patient_status, alarm_status, confidence_score
            
        except Exception as e:
            print(f"Model prediction failed, using rule-based: {e}")
            return self._rule_based_prediction(text)
    
    def _rule_based_prediction(self, text):
        """Rule-based fallback prediction"""
        text_lower = text.lower()
        
        high_keywords = ['chest pain', 'heart attack', 'stroke', 'severe', 'critical', 'bleeding', 'unconscious']
        medium_keywords = ['fever', 'headache', 'dizziness', 'moderate', 'pain', 'vomiting']
        
        high_count = sum(1 for keyword in high_keywords if keyword in text_lower)
        medium_count = sum(1 for keyword in medium_keywords if keyword in text_lower)
        
        if high_count > 0:
            urgency = "High"
            confidence = 0.85
        elif medium_count > 0:
            urgency = "Medium" 
            confidence = 0.75
        else:
            urgency = "Low"
            confidence = 0.65
        
        patient_status, alarm_status = self._get_patient_status(urgency, confidence)
        return urgency, patient_status, alarm_status, confidence
    
    def _get_patient_status(self, urgency, confidence):
        """Determine patient status and alarm based on urgency"""
        if urgency == "High":
            return "Critical", "Notified to Dr"
        elif urgency == "Medium":
            return "Stable", "Monitoring"
        else:
            return "Good", "Routine"