import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class MedicalBERTPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.urgency_labels = ['Low', 'Medium', 'High']
        self.load_model()
    
    def load_model(self):
        """Load the trained BERT model and tokenizer"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, local_files_only=True, num_labels=3
            )
            self.model.eval()
            logger.info("✅ Medical BERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load medical model: {e}")
            raise
    
    def predict_urgency_with_confidence(self, text):
        """Predict urgency with confidence scores"""
        if not text or len(text.strip()) < 5:
            return "Low", "Insufficient information", "Not Notified", 0.5
        
        try:
            # Tokenize and predict
            inputs = self.tokenizer(
                text, truncation=True, padding=True, max_length=512, return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_idx = torch.argmax(predictions, dim=1).item()
                confidence = predictions[0][predicted_class_idx].item()
            
            urgency_level = self.urgency_labels[predicted_class_idx]
            patient_status, alarm_status = self._determine_status_and_alarm(urgency_level, text, confidence)
            
            return urgency_level, patient_status, alarm_status, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback with medium confidence
            urgency_level = self.keyword_detection(text)
            patient_status, alarm_status = self._determine_status_and_alarm(urgency_level, text, 0.5)
            return urgency_level, patient_status, alarm_status, 0.5
    
    def predict_urgency(self, text):
        """Backward compatibility"""
        urgency, status, alarm, confidence = self.predict_urgency_with_confidence(text)
        return urgency, status, alarm
    
    def _determine_status_and_alarm(self, urgency_level, text, confidence):
        """Determine patient status and alarm"""
        text_lower = text.lower()
        
        if urgency_level == "High":
            if any(word in text_lower for word in ['chest pain', 'heart', 'cardiac']):
                status = f"Critical cardiac condition (confidence: {confidence:.2f})"
            elif any(word in text_lower for word in ['breathing', 'choking', 'respiratory']):
                status = f"Respiratory emergency (confidence: {confidence:.2f})"
            else:
                status = f"Critical condition (confidence: {confidence:.2f})"
            alarm = "Notified to Dr"
            
        elif urgency_level == "Medium":
            status = f"Moderate condition (confidence: {confidence:.2f})"
            alarm = "Not Notified"
            
        else:
            status = f"Stable condition (confidence: {confidence:.2f})"
            alarm = "Not Notified"
        
        return status, alarm
    
    def keyword_detection(self, text):
        """Keyword-based urgency detection"""
        if not text:
            return "Low"
            
        text_lower = text.lower()
        
        high_urgency_keywords = [
            'chest pain', 'heart attack', 'stroke', 'difficulty breathing',
            'unconscious', 'severe bleeding', 'choking', 'cardiac arrest'
        ]
        
        medium_urgency_keywords = [
            'fever', 'headache', 'vomiting', 'abdominal pain', 'dizziness',
            'infection', 'pain', 'fracture'
        ]
        
        for keyword in high_urgency_keywords:
            if keyword in text_lower:
                return "High"
        
        for keyword in medium_urgency_keywords:
            if keyword in text_lower:
                return "Medium"
        
        return "Low"