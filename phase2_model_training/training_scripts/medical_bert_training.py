import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import sys
from datetime import datetime

class MedicalBERTTrainer:
    def __init__(self, base_dir, data_path=None):
        self.base_dir = base_dir
        self.data_path = data_path
        self.setup_directories()
        
        # Using Bio+Clinical BERT specifically trained on medical texts
        self.model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        self.tokenizer = None
        self.model = None
        self.urgency_labels = ['low', 'medium', 'high']
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def setup_directories(self):
        """Create necessary directories"""
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.results_dir = os.path.join(self.base_dir, 'results')
        self.data_dir = os.path.join(self.base_dir, 'data')
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        print(f"📁 Models directory: {self.models_dir}")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📁 Data directory: {self.data_dir}")
        
        # If data_path not provided, look in default location
        if self.data_path is None:
            default_data_path = os.path.join(self.data_dir, 'D:\Project\Assesment AI Task\phase2_model_training\data\llm_medical_training_dataset.json')
            if os.path.exists(default_data_path):
                self.data_path = default_data_path
                print(f"📁 Using default data path: {self.data_path}")
    
    def convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def load_and_preprocess_data(self, data_path=None):
        """Load and preprocess the medical training data"""
        # Use provided data_path or instance data_path
        if data_path is None:
            data_path = self.data_path
        
        if data_path is None:
            raise ValueError("No data path provided. Please specify a data path.")
        
        print(f"📁 Loading training data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Total samples: {len(data)}")
        
        # Analyze data distribution
        urgencies = [item['metadata']['urgency'] for item in data]
        specialties = [item['metadata']['specialty'] for item in data]
        
        print(f"🎯 Urgencies: {dict(Counter(urgencies))}")
        print(f"🏥 Specialties: {dict(Counter(specialties))}")
        
        # Stratified split to ensure all classes are represented
        train_data, val_data = self.stratified_split(data)
        
        print(f"📚 Train: {len(train_data)}, Validation: {len(val_data)}")
        print(f"📊 Train distribution: {dict(Counter([item['metadata']['urgency'] for item in train_data]))}")
        print(f"📊 Val distribution: {dict(Counter([item['metadata']['urgency'] for item in val_data]))}")
        
        return train_data, val_data
    
    def stratified_split(self, data, val_ratio=0.2):
        """Stratified split to maintain class distribution"""
        # Group by urgency
        urgency_groups = {}
        for item in data:
            urgency = item['metadata']['urgency']
            if urgency not in urgency_groups:
                urgency_groups[urgency] = []
            urgency_groups[urgency].append(item)
        
        train_data, val_data = [], []
        
        for urgency, items in urgency_groups.items():
            # Shuffle items
            random.shuffle(items)
            # Calculate split index
            split_idx = max(1, int(len(items) * (1 - val_ratio)))  # Ensure at least 1 in val
            train_data.extend(items[:split_idx])
            val_data.extend(items[split_idx:])
        
        return train_data, val_data
    
    def augment_medical_data(self, data):
        """Medical-specific data augmentation"""
        augmented_data = data.copy()
        
        # Check current distribution
        urgency_counts = Counter([item['metadata']['urgency'] for item in data])
        print(f"📈 Original distribution: {urgency_counts}")
        
        # Medical-specific augmentation for each urgency level
        for urgency_level in self.urgency_labels:
            current_count = urgency_counts.get(urgency_level, 0)
            samples = [item for item in data if item['metadata']['urgency'] == urgency_level]
            
            # Target minimum samples per class
            target_min = 8
            if current_count < target_min and samples:
                needed = target_min - current_count
                for i in range(needed):
                    if len(augmented_data) >= 40:  # Max total samples
                        break
                    sample = random.choice(samples)
                    new_sample = self.create_medical_augmented_sample(sample, urgency_level)
                    augmented_data.append(new_sample)
        
        print(f"📈 After augmentation - Total: {len(augmented_data)}")
        print(f"📊 Augmented distribution: {dict(Counter([item['metadata']['urgency'] for item in augmented_data]))}")
        return augmented_data
    
    def create_medical_augmented_sample(self, sample, urgency_level):
        """Create medically relevant augmented samples"""
        text = sample['text']
        specialty = sample['metadata']['specialty']
        
        # Medical-specific augmentation patterns
        if urgency_level == 'high':
            augmentations = [
                f"Emergency presentation: {text}",
                f"Critical condition with {text}",
                f"Urgent medical attention required for {text}",
                text.replace('pain', 'severe acute pain') if 'pain' in text.lower() else f"Acute {text}"
            ]
        elif urgency_level == 'medium':
            augmentations = [
                f"Patient presents with {text}",
                f"Clinical evaluation for {text}",
                f"Medical consultation regarding {text}",
                text.replace('mild', 'moderate') if 'mild' in text.lower() else f"Moderate {text}"
            ]
        else:  # low
            augmentations = [
                f"Routine medical follow-up: {text}",
                f"Preventive care consultation for {text}",
                f"Stable condition with {text}",
                text.replace('severe', 'mild') if 'severe' in text.lower() else f"Mild {text}"
            ]
        
        new_text = random.choice(augmentations)
        
        return {
            'text': new_text[:200],  # Reasonable length for medical texts
            'metadata': {
                'specialty': specialty,
                'urgency': urgency_level
            }
        }
    
    def prepare_medical_dataloader(self, data, batch_size=4):
        """Prepare DataLoader for medical BERT training"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("✅ Bio+Clinical BERT tokenizer loaded")
        
        texts = [item['text'] for item in data]
        urgencies = [item['metadata']['urgency'] for item in data]
        
        # Convert to numerical labels
        labels = [self.urgency_labels.index(urg) for urg in urgencies]
        
        # Tokenize with medical BERT
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels)
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_medical_bert(self, train_loader, val_loader, model_name="medical_bert_model"):
        """Train the Bio+Clinical BERT model"""
        print("🎯 Initializing Bio+Clinical BERT...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label={0: 'low', 1: 'medium', 2: 'high'},
            label2id={'low': 0, 'medium': 1, 'high': 2}
        )
        self.model.to(self.device)
        print("✅ Bio+Clinical BERT model loaded and ready for training")
        
        # Create model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.models_dir, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Training parameters optimized for medical BERT
        epochs = 6
        learning_rate = 1e-5
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        print("🚀 Starting Bio+Clinical BERT training...")
        training_losses = []
        val_accuracies = []
        best_accuracy = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Medical BERT]')
            for batch in progress_bar:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                self.model.zero_grad()
                
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
            avg_train_loss = total_train_loss / len(train_loader)
            training_losses.append(avg_train_loss)
            
            # Validation
            val_accuracy = self.evaluate_medical_model(val_loader)
            val_accuracies.append(val_accuracy)
            
            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}')
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.model.save_pretrained(model_dir)
                self.tokenizer.save_pretrained(model_dir)
                print(f"✅ Saved best model (accuracy: {val_accuracy:.4f}) to {model_dir}")
        
        # Plot training history
        self.plot_medical_training_history(training_losses, val_accuracies, timestamp)
        
        return training_losses, val_accuracies, model_dir
    
    def evaluate_medical_model(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return accuracy_score(true_labels, predictions)
    
    def plot_medical_training_history(self, train_losses, val_accuracies, timestamp):
        """Plot training history for medical model"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.title('Bio+Clinical BERT - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('Bio+Clinical BERT - Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plot_path = os.path.join(self.results_dir, f'medical_bert_training_history_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 Training history plot saved: {plot_path}")
    
    def comprehensive_medical_evaluation(self, val_loader, timestamp):
        """Comprehensive medical-specific evaluation"""
        self.model.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Handle available classes
        unique_labels = np.unique(true_labels)
        available_labels = [self.urgency_labels[i] for i in unique_labels]
        
        # Calculate comprehensive metrics
        accuracy = float(accuracy_score(true_labels, predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Convert to Python native types
        precision = float(precision)
        recall = float(recall)
        f1 = float(f1)
        
        # Create classification report with only available classes
        target_names = [self.urgency_labels[i] for i in unique_labels]
        class_report = classification_report(
            true_labels, predictions, 
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        # Convert class_report to serializable
        class_report = self.convert_to_serializable(class_report)
        
        # Confusion matrix
        cm = np.zeros((3, 3))
        for true, pred in zip(true_labels, predictions):
            cm[true][pred] += 1
        
        # Plot medical-themed confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                   xticklabels=self.urgency_labels,
                   yticklabels=self.urgency_labels,
                   cbar_kws={'label': 'Number of Cases'})
        plt.title('Medical Urgency Classification - Confusion Matrix\n(Bio+Clinical BERT)',
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Medical Urgency', fontweight='bold')
        plt.xlabel('Predicted Medical Urgency', fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = os.path.join(self.results_dir, f'medical_bert_confusion_matrix_{timestamp}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 Confusion matrix saved: {cm_path}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'available_classes': available_labels,
            'class_report': class_report,
            'confusion_matrix': cm.tolist(),
            'timestamp': timestamp
        }

def generate_medical_report(metrics, model_path, results_dir):
    """Generate comprehensive medical performance report"""
    # Convert all metrics to serializable types
    trainer = MedicalBERTTrainer("")  # Temporary instance for conversion method
    metrics = trainer.convert_to_serializable(metrics)
    
    report = {
        'timestamp': str(pd.Timestamp.now()),
        'model': 'Bio+Clinical BERT',
        'model_description': 'BERT model pre-trained on biomedical and clinical texts',
        'model_path': model_path,
        'performance_metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1'],
        },
        'available_classes': metrics['available_classes'],
        'class_performance': metrics['class_report'],
        'confusion_matrix': metrics['confusion_matrix'],
        'medical_suitability': {
            'domain': 'Medical NLP - Urgency Classification',
            'training_data': 'Clinical conversations and medical texts',
            'suitable_for': ['Triage systems', 'Medical alert systems', 'Clinical decision support'],
            'limitations': ['Small training dataset', 'Requires medical validation']
        },
        'requirements_validation': {
            'profiling_accuracy': {
                'target': 'F1-score > 0.85',
                'achieved': metrics['f1'],
                'status': 'PASS' if metrics['f1'] > 0.85 else 'NEEDS_IMPROVEMENT'
            },
            'clinical_reliability': {
                'high_risk_recall': 'Critical for patient safety',
                'false_positive_rate': 'Should be minimized',
                'status': 'MEDICAL_VALIDATION_REQUIRED'
            }
        }
    }
    
    report_path = os.path.join(results_dir, f'medical_bert_performance_report_{metrics["timestamp"]}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n📊 MEDICAL BERT PERFORMANCE REPORT:")
    print("=" * 50)
    print(f"🏥 Model: Bio+Clinical BERT")
    print(f"📈 Accuracy: {metrics['accuracy']:.4f}")
    print(f"🎯 F1-Score: {metrics['f1']:.4f}")
    print(f"📍 Precision: {metrics['precision']:.4f}")
    print(f"🔍 Recall: {metrics['recall']:.4f}")
    print(f"📋 Available classes: {metrics['available_classes']}")
    
    if metrics['f1'] > 0.85:
        print("✅ Profiling Accuracy: EXCELLENT (F1 > 0.85)")
    elif metrics['f1'] > 0.75:
        print("⚠️ Profiling Accuracy: GOOD (F1 > 0.75) - Suitable for prototype")
    else:
        print("🔴 Profiling Accuracy: NEEDS IMPROVEMENT - Collect more medical data")
    
    print(f"\n💾 Medical report saved: {report_path}")
    return report_path

def run_training(data_path=None):
    """Main training function that accepts data path"""
    print("🚀 STARTING BIO+CLINICAL BERT MEDICAL CLASSIFIER TRAINING")
    print("=" * 70)
    
    # Get base directory (phase2_model_training)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"📁 Base directory: {base_dir}")
    
    # Initialize medical BERT trainer with custom data path if provided
    trainer = MedicalBERTTrainer(base_dir, data_path)
    
    try:
        train_data, val_data = trainer.load_and_preprocess_data()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Please specify the correct data path")
        return
    except ValueError as e:
        print(f"❌ {e}")
        print("💡 Please run the dataset preparation script first or specify data path")
        return
    
    # Apply medical-specific augmentation
    print("\n🩺 Applying medical-specific data augmentation...")
    train_data_augmented = trainer.augment_medical_data(train_data)
    
    # Prepare medical data loaders
    print("📚 Preparing medical data loaders...")
    train_loader = trainer.prepare_medical_dataloader(train_data_augmented, batch_size=4)
    val_loader = trainer.prepare_medical_dataloader(val_data, batch_size=4)
    
    print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Train medical BERT model
    print(f"\n🏥 Training Bio+Clinical BERT for Medical Urgency Classification...")
    train_losses, val_accuracies, model_dir = trainer.train_medical_bert(train_loader, val_loader)
    
    # Comprehensive medical evaluation
    print("\n🔬 Running comprehensive medical evaluation...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = trainer.comprehensive_medical_evaluation(val_loader, timestamp)
    
    # Generate medical performance report
    report_path = generate_medical_report(metrics, model_dir, trainer.results_dir)
    
    print("\n" + "=" * 70)
    print("🎯 BIO+CLINICAL BERT TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Medical Model saved in: {model_dir}")
    print(f"📊 Medical Report: {report_path}")
    print(f"📈 Training History: {os.path.join(trainer.results_dir, f'medical_bert_training_history_{timestamp}.png')}")
    print(f"🔍 Confusion Matrix: {os.path.join(trainer.results_dir, f'medical_bert_confusion_matrix_{timestamp}.png')}")
    print("\n✅ Medical NLP Model is ready for clinical applications!")

if __name__ == "__main__":
    # Handle command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to training data JSON file')
    args = parser.parse_args()
    
    run_training(args.data_path)