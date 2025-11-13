import pandas as pd
import json
import re
import spacy
from collections import Counter
import os
import sys

class MedicalDatasetPreprocessor:
    def __init__(self):
        # Setup paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.base_dir)  # Go up one level to project root
        self.data_dir = os.path.join(self.project_root, 'data')
        self.metadata_dir = os.path.join(self.data_dir, 'metadata')
        self.output_dir = os.path.join(self.base_dir, 'prepared_data')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Dataset path
        self.dataset_path = os.path.join(self.metadata_dir, 'D:\Project\Assesment AI Task\phase1_data_processing\data\metadata\dataset_summary.csv')
        
        print("🔧 Medical Dataset Preprocessor Initialized")
        print(f"📁 Project Root: {self.project_root}")
        print(f"📁 Data Directory: {self.data_dir}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"📄 Dataset Path: {self.dataset_path}")
        
        # Load spaCy model
        self.load_spacy_model()
        
        # Define classification rules
        self.setup_classification_rules()
    
    def load_spacy_model(self):
        """Load spaCy model with fallback"""
        print("🔄 Loading spaCy model...")
        try:
            # Try medical model first
            self.nlp = spacy.load("en_core_sci_md")
            print("✅ Medical spaCy model loaded successfully!")
        except OSError:
            try:
                # Fallback to regular English model
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ Standard English spaCy model loaded (medical model not found)")
                print("💡 For better medical entity recognition, install:")
                print("   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz")
            except OSError:
                print("❌ No spaCy model found. Please install:")
                print("   python -m spacy download en_core_web_sm")
                sys.exit(1)
    
    def setup_classification_rules(self):
        """Setup urgency classification and specialty prediction rules"""
        self.URGENCY_RULES = {
            'high': [
                # Cardiac emergencies
                r'\b(severe|crushing|sharp) chest pain\b',
                r'\bheart attack\b',
                r'\bcardiac arrest\b',
                r'\bshortness of breath\b',
                r'\bdifficulty breathing\b',
                r'\btrouble breathing\b',
                r'\blightheaded\b',
                r'\bfainting\b',
                r'\bloss of consciousness\b',
                r'\bpassing out\b',
                r'\bradiating pain\b',
                r'\bpain spreading\b',
                r'\bsevere pain\b',
                r'\b10/10 pain\b',
                r'\b9/10 pain\b',
                r'\b8/10 pain\b',
                r'\bworst pain\b',
                
                # Other emergencies
                r'\bsevere bleeding\b',
                r'\buncontrolled bleeding\b',
                r'\bstroke\b',
                r'\bnumbness\b',
                r'\bweakness\b',
                r'\bparalysis\b',
                r'\bvision loss\b',
                r'\bsevere headache\b',
                r'\bseizure\b'
            ],
            'medium': [
                r'\bchest pain\b',
                r'\bfever\b',
                r'\bpersistent cough\b',
                r'\bworsening symptoms\b',
                r'\babdominal pain\b',
                r'\bvomiting\b',
                r'\bdiarrhea\b',
                r'\bdehydration\b',
                r'\bmoderate pain\b',
                r'\b7/10 pain\b',
                r'\b6/10 pain\b',
                r'\binfection\b',
                r'\binflammatory\b',
                r'\bswelling\b',
                r'\bredness\b',
                r'\bdizziness\b',
                r'\bnausea\b'
            ],
            'low': [
                r'\broutine\b',
                r'\bfollow.up\b',
                r'\bcheck.up\b',
                r'\bmanagement\b',
                r'\breview\b',
                r'\bmild pain\b',
                r'\bchronic condition\b',
                r'\bstable\b',
                r'\bpreventive care\b',
                r'\bvaccination\b',
                r'\bscreening\b',
                r'\bcold symptoms\b',
                r'\bmild cough\b',
                r'\brunny nose\b',
                r'\b1/10 pain\b',
                r'\b2/10 pain\b',
                r'\b3/10 pain\b'
            ]
        }
        
        self.SPECIALTY_KEYWORDS = {
            'Cardiology': [
                'chest', 'heart', 'cardiac', 'breathing', 'palpitations', 
                'blood pressure', 'hypertension', 'cholesterol', 'ecg'
            ],
            'Gastroenterology': [
                'stomach', 'abdominal', 'vomiting', 'diarrhea', 'nausea', 
                'bowel', 'digestive', 'constipation', 'indigestion'
            ],
            'Musculoskeletal': [
                'pain', 'joint', 'muscle', 'elbow', 'shoulder', 'knee', 
                'back', 'swelling', 'tendon', 'ligament', 'fracture'
            ],
            'Dermatology': [
                'rash', 'skin', 'itching', 'redness', 'lesion', 'acne', 
                'eczema', 'dermatitis', 'psoriasis'
            ],
            'Respiratory': [
                'cough', 'breathing', 'wheezing', 'lungs', 'respiratory', 
                'asthma', 'pneumonia', 'bronchitis'
            ],
            'General Medicine': [
                'fever', 'fatigue', 'general', 'routine', 'check.up'
            ]
        }
    
    def load_dataset(self):
        """Load the dataset from CSV file"""
        print(f"📁 Loading dataset from: {self.dataset_path}")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset file not found: {self.dataset_path}")
            print("💡 Please check the file path and ensure dataset_summary.csv exists")
            return None
        
        try:
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Dataset loaded successfully: {len(df)} records")
            print(f"📊 Columns: {list(df.columns)}")
            
            # Display basic info
            if 'specialty' in df.columns:
                print(f"📈 Specialty distribution:")
                print(df['specialty'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def extract_key_information(self, text):
        """Extract key medical information using spaCy"""
        if pd.isna(text) or text == '':
            return {'symptoms': [], 'conditions': [], 'severity_indicators': []}
        
        doc = self.nlp(text.lower())
        
        # Extract medical entities
        symptoms = []
        conditions = []
        severity_indicators = []
        
        for ent in doc.ents:
            if ent.label_ in ["DISEASE", "SYMPTOM", "SIGN", "PROBLEM", "CONDITION"]:
                if any(word in ent.text.lower() for word in ['severe', 'mild', 'moderate', 'pain', 'ache']):
                    symptoms.append(ent.text)
                else:
                    conditions.append(ent.text)
        
        # Extract severity words
        severity_words = ['severe', 'mild', 'moderate', 'sharp', 'chronic', 'acute', 'worsening', 'unbearable']
        for token in doc:
            if token.text in severity_words:
                severity_indicators.append(token.text)
        
        # Extract pain levels
        pain_levels = re.findall(r'(\d+)/10 pain', text.lower())
        severity_indicators.extend([f"{level}/10 pain" for level in pain_levels])
        
        return {
            'symptoms': list(set(symptoms)),
            'conditions': list(set(conditions)),
            'severity_indicators': list(set(severity_indicators))
        }
    
    def classify_urgency(self, text, extracted_info):
        """Classify urgency based on rules and extracted information"""
        if pd.isna(text) or text == '':
            return 'low'
        
        text_lower = text.lower()
        
        # Check high urgency rules
        for pattern in self.URGENCY_RULES['high']:
            if re.search(pattern, text_lower):
                return 'high'
        
        # Check medium urgency rules
        for pattern in self.URGENCY_RULES['medium']:
            if re.search(pattern, text_lower):
                return 'medium'
        
        # Check low urgency rules
        for pattern in self.URGENCY_RULES['low']:
            if re.search(pattern, text_lower):
                return 'low'
        
        # Default based on severity indicators
        if any(word in text_lower for word in ['severe', 'emergency', 'urgent', 'critical']):
            return 'high'
        elif any(word in text_lower for word in ['moderate', 'worsening', 'persistent']):
            return 'medium'
        else:
            return 'low'
    
    def predict_specialty(self, text, extracted_info):
        """Predict medical specialty based on keywords"""
        if pd.isna(text) or text == '':
            return 'General Medicine'
        
        text_lower = text.lower()
        specialty_scores = {}
        
        for specialty, keywords in self.SPECIALTY_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of keyword
                score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            specialty_scores[specialty] = score
        
        # Return specialty with highest score
        predicted_specialty = max(specialty_scores, key=specialty_scores.get)
        
        # If no strong match, use General Medicine
        if specialty_scores[predicted_specialty] == 0:
            return 'General Medicine'
        
        return predicted_specialty
    
    def create_concise_summary(self, text, extracted_info, urgency, specialty):
        """Create a concise summary in the required format"""
        if pd.isna(text) or text == '':
            return "No transcription available"
        
        symptoms = extracted_info['symptoms']
        severity = extracted_info['severity_indicators']
        
        # Build the summary text
        if symptoms:
            main_symptoms = symptoms[:3]  # Take first 3 symptoms
            symptom_text = ", ".join(main_symptoms)
            
            if severity:
                severity_text = severity[0]  # Take the most prominent severity indicator
                summary = f"Patient experiencing {severity_text} {symptom_text}"
            else:
                summary = f"Patient experiencing {symptom_text}"
        else:
            # Fallback: use keywords from text
            if 'chest pain' in text.lower():
                summary = "Patient experiencing chest pain"
            elif 'fever' in text.lower() and 'cough' in text.lower():
                summary = "Patient with fever and persistent cough"
            elif 'pain' in text.lower():
                summary = "Patient reporting pain symptoms"
            else:
                # Use first 100 characters as fallback
                summary = text[:100] + "..." if len(text) > 100 else text
        
        # Add duration if mentioned
        duration_pattern = r'(\d+\s*(?:hour|day|week|month)s?)'
        duration_match = re.search(duration_pattern, text.lower())
        if duration_match:
            summary += f" for {duration_match.group(1)}"
        
        # Add urgency context
        if urgency == 'high':
            summary += " with emergency symptoms"
        elif urgency == 'medium':
            summary += " with concerning symptoms"
        else:
            summary += " for evaluation"
        
        return summary
    
    def process_dataset(self, df):
        """Process the entire dataset and create training data"""
        print("🔄 Processing medical conversations...")
        
        training_data = []
        processed_count = 0
        
        for idx, row in df.iterrows():
            try:
                text = row['transcription'] if 'transcription' in row else ''
                original_specialty = row['specialty'] if 'specialty' in row else 'Unknown'
                filename = row['filename'] if 'filename' in row else f"record_{idx}"
                
                if processed_count % 10 == 0:  # Progress indicator
                    print(f"   Processed {processed_count}/{len(df)} records...")
                
                # Extract information using spaCy
                extracted_info = self.extract_key_information(text)
                
                # Classify urgency
                urgency = self.classify_urgency(text, extracted_info)
                
                # Predict specialty
                specialty = self.predict_specialty(text, extracted_info)
                
                # Create concise summary
                concise_text = self.create_concise_summary(text, extracted_info, urgency, specialty)
                
                # Create the training example
                training_example = {
                    "text": concise_text,
                    "metadata": {
                        "specialty": specialty,
                        "urgency": urgency,
                        "original_filename": filename,
                        "original_specialty": original_specialty
                    }
                }
                
                training_data.append(training_example)
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing record {idx}: {e}")
                continue
        
        return training_data
    
    def save_datasets(self, training_data):
        """Save the processed datasets in multiple formats"""
        print("💾 Saving processed datasets...")
        
        # Save as JSON (pretty format)
        json_output = os.path.join(self.output_dir, 'llm_medical_training_dataset.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON dataset saved: {json_output}")
        
        # Save as JSONL (for LLM training)
        jsonl_output = os.path.join(self.output_dir, 'llm_medical_training_dataset.jsonl')
        with open(jsonl_output, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print(f"✅ JSONL dataset saved: {jsonl_output}")
        
        # Save as CSV for analysis
        csv_data = []
        for example in training_data:
            csv_data.append({
                'text': example['text'],
                'specialty': example['metadata']['specialty'],
                'urgency': example['metadata']['urgency'],
                'original_filename': example['metadata']['original_filename'],
                'original_specialty': example['metadata']['original_specialty']
            })
        
        csv_output = os.path.join(self.output_dir, 'llm_medical_training_dataset.csv')
        pd.DataFrame(csv_data).to_csv(csv_output, index=False)
        print(f"✅ CSV dataset saved: {csv_output}")
        
        # Save sample preview
        sample_output = os.path.join(self.output_dir, 'sample_training_data.json')
        with open(sample_output, 'w', encoding='utf-8') as f:
            json.dump(training_data[:10], f, indent=2, ensure_ascii=False)
        print(f"✅ Sample data saved: {sample_output}")
        
        return json_output, jsonl_output, csv_output
    
    def generate_statistics(self, training_data):
        """Generate and display dataset statistics"""
        print("\n📊 DATASET STATISTICS")
        print("=" * 50)
        
        urgency_counts = Counter([item['metadata']['urgency'] for item in training_data])
        specialty_counts = Counter([item['metadata']['specialty'] for item in training_data])
        original_specialty_counts = Counter([item['metadata']['original_specialty'] for item in training_data])
        
        print(f"📈 Total Examples: {len(training_data)}")
        
        print(f"\n🎯 Urgency Distribution:")
        for urgency, count in urgency_counts.most_common():
            percentage = (count / len(training_data)) * 100
            print(f"   {urgency.upper():8}: {count:3d} examples ({percentage:5.1f}%)")
        
        print(f"\n🏥 Predicted Specialty Distribution:")
        for specialty, count in specialty_counts.most_common():
            percentage = (count / len(training_data)) * 100
            print(f"   {specialty:20}: {count:3d} examples ({percentage:5.1f}%)")
        
        print(f"\n📋 Original Specialty Distribution:")
        for specialty, count in original_specialty_counts.most_common():
            percentage = (count / len(training_data)) * 100
            print(f"   {specialty:20}: {count:3d} examples ({percentage:5.1f}%)")
        
        # Save statistics
        stats = {
            'total_examples': len(training_data),
            'urgency_distribution': dict(urgency_counts),
            'specialty_distribution': dict(specialty_counts),
            'original_specialty_distribution': dict(original_specialty_counts)
        }
        
        stats_output = os.path.join(self.output_dir, 'dataset_statistics.json')
        with open(stats_output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Statistics saved: {stats_output}")
    
    def run(self):
        """Main method to run the entire preprocessing pipeline"""
        print("🚀 STARTING MEDICAL DATASET PREPROCESSING")
        print("=" * 60)
        
        # Load dataset
        df = self.load_dataset()
        if df is None:
            return
        
        # Process dataset
        training_data = self.process_dataset(df)
        
        if not training_data:
            print("❌ No training data was processed")
            return
        
        # Save datasets
        self.save_datasets(training_data)
        
        # Generate statistics
        self.generate_statistics(training_data)
        
        # Display samples
        print("\n👀 SAMPLE TRAINING EXAMPLES:")
        print("=" * 50)
        for i, example in enumerate(training_data[:5]):
            print(f"{i+1}. Text: {example['text']}")
            print(f"   Specialty: {example['metadata']['specialty']}")
            print(f"   Urgency: {example['metadata']['urgency']}")
            print(f"   Original: {example['metadata']['original_specialty']}")
            print()
        
        print("🎉 DATASET PREPARATION COMPLETED SUCCESSFULLY!")
        print(f"📁 All files saved in: {self.output_dir}")

def main():
    """Main function to run the preprocessor"""
    preprocessor = MedicalDatasetPreprocessor()
    preprocessor.run()

if __name__ == "__main__":
    main()