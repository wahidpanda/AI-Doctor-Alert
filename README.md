now for all task i have to write redme.md file 

app name :

AI-Doctor-Alert
main i did:

first collect medical voice notes from public repositroy, then take data for audio to text convertion, after that from text analysze keyword detection and dataset ready, after that model training, evaluation after that, web application, alert system implement 

this is my methodology
so make a workflow diagram in this with code

now phase wise task
we first run in all process testing in kaggle notebook, (use kaggle logo here)
link: https://www.kaggle.com/code/islamwahid/ai-assesment-medical-alert

1st phas

- data extraction load, transfrom pipline create 
pipeleine code:
lass DataProcessingPipeline:
    def __init__(self):
        self.downloader = KaggleDatasetDownloader()
        self.collector = AudioDataCollector()
        self.preprocessor = AudioPreprocessor()
        self.transcriber = AudioTranscriber()
        self.analyzer = DataAnalyzer()
        self.logger = logging.getLogger(__name__)
so write from here ,

after that save data for 2nd phase 

in 2nd phase:



