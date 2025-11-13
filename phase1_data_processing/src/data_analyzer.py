import pandas as pd
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config.paths import DATA_PATHS

logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.df = None
    
    def load_transcripts(self, metadata_file=None):
        """Load processed transcripts with proper error handling"""
        if metadata_file is None:
            metadata_file = DATA_PATHS['metadata'] / 'complete_metadata.json'
        
        try:
            if not metadata_file.exists():
                logger.warning(f"Metadata file not found: {metadata_file}")
                return None
            
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                logger.warning("Metadata file is empty")
                return None
            
            self.df = pd.DataFrame(data)
            logger.info(f"Loaded {len(self.df)} transcripts")
            return self.df
        except Exception as e:
            logger.error(f"Error loading transcripts: {e}")
            return None
    
    def analyze_dataset(self):
        """Comprehensive dataset analysis with safe column access"""
        if self.df is None:
            if not self.load_transcripts():
                return {
                    'total_files': 0,
                    'average_confidence': 0,
                    'total_words': 0,
                    'specialty_distribution': {},
                    'confidence_by_specialty': {},
                    'average_word_count': 0,
                    'word_count_stats': {}
                }
        
        analysis = {
            'total_files': len(self.df),
            'average_confidence': 0,
            'total_words': 0,
            'specialty_distribution': {},
            'confidence_by_specialty': {},
            'average_word_count': 0,
            'word_count_stats': {}
        }
        
        # Safely calculate metrics
        if 'transcription_confidence' in self.df.columns:
            analysis['average_confidence'] = self.df['transcription_confidence'].mean()
        
        if 'transcription' in self.df.columns:
            self.df['word_count'] = self.df['transcription'].apply(lambda x: len(str(x).split()))
            analysis['total_words'] = self.df['word_count'].sum()
            analysis['average_word_count'] = self.df['word_count'].mean()
            analysis['word_count_stats'] = self.df['word_count'].describe().to_dict()
        
        if 'specialty' in self.df.columns:
            analysis['specialty_distribution'] = self.df['specialty'].value_counts().to_dict()
            
            if 'transcription_confidence' in self.df.columns:
                analysis['confidence_by_specialty'] = self.df.groupby('specialty')['transcription_confidence'].mean().to_dict()
        
        return analysis
    
    def generate_reports(self):
        """Generate analysis reports and visualizations with error handling"""
        analysis = self.analyze_dataset()
        
        # Save analysis report
        report_file = DATA_PATHS['analysis_reports'] / 'dataset_analysis.json'
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved analysis report: {report_file}")
        
        # Only generate visualizations if we have data
        if analysis['total_files'] > 0 and self.df is not None:
            self._generate_visualizations()
        else:
            logger.warning("No data available for visualizations")
        
        # Create LLM training data
        training_data = self._create_llm_training_data()
        
        logger.info(f"Analysis reports saved to {DATA_PATHS['analysis_reports']}")
        return analysis
    
    def _generate_visualizations(self):
        """Generate data visualization plots with error handling"""
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Specialty distribution
            if 'specialty' in self.df.columns:
                specialty_counts = self.df['specialty'].value_counts()
                axes[0, 0].bar(specialty_counts.index, specialty_counts.values)
                axes[0, 0].set_title('Medical Specialty Distribution')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Confidence distribution
            if 'transcription_confidence' in self.df.columns:
                axes[0, 1].hist(self.df['transcription_confidence'], bins=20, alpha=0.7)
                axes[0, 1].set_title('Transcription Confidence Distribution')
                axes[0, 1].set_xlabel('Confidence')
                axes[0, 1].set_ylabel('Frequency')
            
            # Word count distribution
            if 'word_count' in self.df.columns:
                axes[1, 0].hist(self.df['word_count'], bins=20, alpha=0.7)
                axes[1, 0].set_title('Word Count Distribution')
                axes[1, 0].set_xlabel('Word Count')
                axes[1, 0].set_ylabel('Frequency')
            
            # Confidence by specialty
            if 'specialty' in self.df.columns and 'transcription_confidence' in self.df.columns:
                confidence_by_specialty = self.df.groupby('specialty')['transcription_confidence'].mean()
                axes[1, 1].bar(confidence_by_specialty.index, confidence_by_specialty.values)
                axes[1, 1].set_title('Average Confidence by Specialty')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(DATA_PATHS['analysis_reports'] / 'dataset_visualizations.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Generated visualization plots")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _create_llm_training_data(self):
        """Create JSONL format for LLM training with error handling"""
        training_data = []
        
        if self.df is None or len(self.df) == 0:
            logger.warning("No data available for LLM training data creation")
            return training_data
        
        for _, row in self.df.iterrows():
            try:
                training_example = {
                    "text": row.get('transcription', ''),
                    "metadata": {
                        "filename": row.get('filename', ''),
                        "specialty": row.get('specialty', 'Unknown'),
                        "confidence": row.get('transcription_confidence', 0),
                        "word_count": len(str(row.get('transcription', '')).split()),
                        "source": row.get('source', 'unknown'),
                        "is_placeholder": row.get('is_placeholder', False)
                    }
                }
                training_data.append(training_example)
            except Exception as e:
                logger.warning(f"Error processing row for LLM training: {e}")
                continue
        
        # Save as JSONL
        jsonl_file = DATA_PATHS['llm_training'] / 'medical_transcripts.jsonl'
        with open(jsonl_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"LLM training data saved: {jsonl_file} ( {len(training_data)} examples)")
        
        # Also save as CSV for easy viewing
        try:
            csv_file = DATA_PATHS['llm_training'] / 'medical_transcripts.csv'
            export_df = self.df[['filename', 'specialty', 'transcription', 'transcription_confidence']].copy()
            export_df['word_count'] = export_df['transcription'].apply(lambda x: len(str(x).split()))
            export_df.to_csv(csv_file, index=False)
            logger.info(f"CSV training data saved: {csv_file}")
        except Exception as e:
            logger.error(f"Error saving CSV training data: {e}")
        
        return training_data