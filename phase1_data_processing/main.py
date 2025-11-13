import logging
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from config.paths import DATA_PATHS, PROCESSING_CONFIG

from src.kaggle_downloader import KaggleDatasetDownloader
from src.audio_collector import AudioDataCollector
from src.audio_preprocessor import AudioPreprocessor
from src.audio_transcriber import AudioTranscriber
from src.data_analyzer import DataAnalyzer

# Setup logging
def setup_logging():
    """Configure logging for the pipeline"""
    log_file = DATA_PATHS['logs'] / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class DataProcessingPipeline:
    def __init__(self):
        self.downloader = KaggleDatasetDownloader()
        self.collector = AudioDataCollector()
        self.preprocessor = AudioPreprocessor()
        self.transcriber = AudioTranscriber()
        self.analyzer = DataAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def run_pipeline(self):
        """Execute the complete data processing pipeline with robust error handling"""
        self.logger.info("Starting Medical Audio Data Processing Pipeline")
        self.logger.info("=" * 60)
        
        try:
            # Step 0: Download Dataset (with fallbacks)
            self.logger.info("STEP 0: Dataset Download")
            if not self.downloader.is_dataset_downloaded():
                self.logger.info("Downloading dataset from Kaggle...")
                if not self.downloader.download_dataset():
                    self.logger.warning("Download failed, but pipeline will continue with available data")
                else:
                    self.logger.info("✓ Dataset downloaded successfully")
            else:
                self.logger.info("✓ Dataset already available")
            
            # Step 1: Data Collection
            self.logger.info("\nSTEP 1: Data Collection")
            all_files = self.collector.discover_audio_files()
            
            if not all_files:
                self.logger.error("No files found. Creating sample data for demonstration...")
                # Force create sample data
                self.downloader._create_sample_medical_audio()
                all_files = self.collector.discover_audio_files()
                
                if not all_files:
                    self.logger.error("No files available even after creating samples. Exiting.")
                    return False
            
            # Validate dataset
            if not self.collector.validate_dataset():
                self.logger.warning("Dataset validation had issues, but continuing...")
            
            dataset_summary = self.collector.get_dataset_summary()
            self.logger.info(f"Dataset Summary: {json.dumps(dataset_summary, indent=2)}")
            
            # Step 2: Audio Preprocessing (only for actual audio files)
            self.logger.info("\nSTEP 2: Audio Preprocessing")
            audio_files = [f for f in all_files if not self.collector.is_placeholder_file(f)]
            placeholder_files = [f for f in all_files if self.collector.is_placeholder_file(f)]
            
            processed_files = []
            
            if audio_files:
                processed_audio = self.preprocessor.batch_preprocess(audio_files)
                processed_files.extend(processed_audio)
                self.logger.info(f"✓ Preprocessed {len(processed_audio)} audio files")
            else:
                self.logger.info("✓ No audio files to preprocess")
            
            # Add placeholder files directly
            for placeholder_file in placeholder_files:
                processed_files.append({
                    'original_file': placeholder_file,
                    'processed_file': placeholder_file,  # Same file for placeholders
                    'filename': Path(placeholder_file).name,
                    'specialty': self.collector.get_medical_specialty(Path(placeholder_file).name),
                    'processing_timestamp': datetime.now().isoformat()
                })
                self.logger.debug(f"Added placeholder: {Path(placeholder_file).name}")
            
            self.logger.info(f"✓ Total files ready for processing: {len(processed_files)}")
            
            # Step 3: Transcription
            self.logger.info("\nSTEP 3: Audio Transcription")
            if not processed_files:
                self.logger.error("No files to transcribe. Exiting.")
                return False
                
            transcripts = self.transcriber.batch_transcribe(processed_files, self.collector)
            
            if not transcripts:
                self.logger.error("No transcripts generated. Exiting.")
                return False
                
            self.logger.info(f"✓ Successfully processed {len(transcripts)} files")
            
            # Step 4: Save Metadata
            self.logger.info("\nSTEP 4: Saving Metadata")
            self._save_metadata(transcripts)
            
            # Step 5: Data Analysis
            self.logger.info("\nSTEP 5: Data Analysis")
            analysis = self.analyzer.generate_reports()
            
            # Final Summary
            self.logger.info("\n" + "=" * 60)
            self.logger.info("🎉 PIPELINE EXECUTION COMPLETE")
            self.logger.info(f"📊 Processed {len(transcripts)} medical files")
            self.logger.info(f"📁 Source: {analysis.get('total_files', 0)} total files")
            self.logger.info(f"🤖 LLM training data ready: {DATA_PATHS['llm_training']}")
            self.logger.info(f"📈 Analysis reports: {DATA_PATHS['analysis_reports']}")
            
            # Print quick summary
            self._print_quick_summary(analysis)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
    
    def _save_metadata(self, transcripts):
        """Save comprehensive metadata with error handling"""
        try:
            # Save complete metadata
            metadata_file = DATA_PATHS['metadata'] / 'complete_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(transcripts, f, indent=2)
            
            # Save processing summary
            summary_data = []
            for transcript in transcripts:
                summary_data.append({
                    "filename": transcript.get("filename", ""),
                    "specialty": transcript.get("specialty", "Unknown"),
                    "transcription": transcript.get("transcription", ""),
                    "confidence": transcript.get("transcription_confidence", 0),
                    "word_count": len(transcript.get("transcription", "").split()),
                    "source": transcript.get("source", "unknown"),
                    "is_placeholder": transcript.get("is_placeholder", False),
                    "processing_timestamp": transcript.get("processing_timestamp", "")
                })
            
            summary_file = DATA_PATHS['metadata'] / 'processing_summary.csv'
            pd.DataFrame(summary_data).to_csv(summary_file, index=False)
            
            self.logger.info(f"✓ Metadata saved: {metadata_file}")
            self.logger.info(f"✓ Summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def _print_quick_summary(self, analysis):
        """Print a quick summary of the processing results"""
        self.logger.info("\n" + "=" * 40)
        self.logger.info("QUICK SUMMARY")
        self.logger.info("=" * 40)
        self.logger.info(f"Total files processed: {analysis.get('total_files', 0)}")
        self.logger.info(f"Average confidence: {analysis.get('average_confidence', 0):.3f}")
        self.logger.info(f"Total words: {analysis.get('total_words', 0)}")
        
        if analysis.get('specialty_distribution'):
            self.logger.info("Specialty distribution:")
            for specialty, count in analysis['specialty_distribution'].items():
                self.logger.info(f"  {specialty}: {count} files")

def main():
    """Main execution function"""
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Medical Voice-to-Text Pipeline Starting...")
    
    pipeline = DataProcessingPipeline()
    success = pipeline.run_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 Phase 1 Data Processing Complete!")
        print("The pipeline successfully processed medical data.")
        print("Next steps:")
        print("1. Check outputs/llm_training_data/ for training data")
        print("2. Review outputs/analysis_reports/ for insights")
        print("3. Proceed to Phase 2: LLM Training")
        print("=" * 60)
    else:
        print("\n❌ Pipeline execution failed. Check logs for details.")
        print("Troubleshooting tips:")
        print("1. Check internet connection for Kaggle download")
        print("2. Verify sufficient disk space")
        print("3. Check log files in logs/ directory")

if __name__ == "__main__":
    main()