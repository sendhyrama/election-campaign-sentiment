# Request package
import nltk
import matplotlib.pyplot as plt
import re
from sklearn.base import BaseEstimator, TransformerMixin
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, do_stemming=True):
        self.do_stemming = do_stemming

    def fit(self, X, y=None):
        # Tidak ada fitting yang diperlukan, karena kita hanya melakukan preprocessing
        return self

    def transform(self, X):
        # Lakukan preprocessing pada setiap teks dalam X
        return [self.preprocess_text(text) for text in X]

    def preprocess_text(self, Text):
        # Pembersihan teks (cleaning)
        Text = re.sub(r'ð_x[0-9a-fA-F]{4}_x[0-9a-fA-F]{4}x[0-9a-fA-F]{4}', '', Text)
        Text = re.sub(r'â_x0080_x008b', '', Text)
        Text = re.sub(r'ð_x[0-9a-fA-F]{4}_x[0-9a-fA-F]{4}x[0-9a-fA-F]{4}»?', '', Text)
        Text = re.sub(r'(f__x\s*)+', '', Text)
        Text = re.sub(r'@{1,2}[A-Za-z0-9_]+', '', Text)
        Text = re.sub(r'\$\w*', '', Text)
        Text = re.sub(r'^rt[\s]+', '', Text)
        Text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', Text)
        Text = re.sub('&quot;', " ", Text)
        Text = re.sub(r"\d+", " ", str(Text))
        Text = re.sub(r"\b[a-zA-Z]\b", "", str(Text))
        Text = re.sub(r"[^\w\s]", " ", str(Text))
        Text = re.sub(r'(.)\1+', r'\1\1', Text)
        Text = re.sub(r"\s+", " ", str(Text))
        Text = re.sub(r'#', '', Text)
        Text = re.sub(r'[^a-zA-z0-9]', ' ', str(Text))
        Text = re.sub(r'\b\w{1,2}\b', '', Text)
        Text = re.sub(r'\s\s+', ' ', Text)
        Text = re.sub(r'^RT[\s]+', '', Text)
        Text = re.sub(r'^b[\s]+', '', Text)
        Text = re.sub(r'^link[\s]+', '', Text)
        Text = re.sub(r'@\w+', '', Text)
        Text = re.sub('<[^>]+>', '', Text)
        Text = re.sub(r'[\U0001F600-\U0001F64F]', '', Text)
        Text = re.sub(r'quot|amp', '', Text)
        Text = re.sub(r'\b\w*(k\w*w|w\w*k)\w*\b', '', Text)

        # Case folding: lowercase
        Text = Text.lower()

        # Stemming jika diinginkan
        if self.do_stemming:
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            Text = stemmer.stem(Text)

        return Text

# Custom transformer untuk pengecekan sentiment setelah sarkasme
class SentimentAfterSarcasmTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sentiment_column='sentiment_result', sarcasm_column='sarcasm_pred'):
        self.sentiment_column = sentiment_column
        self.sarcasm_column = sarcasm_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Pastikan X adalah list atau array yang memiliki pasangan [sentiment_result, sarcasm_pred]
        sentiment_result, sarcasm_pred = X

        # Indeks dengan sentimen positif
        positive_sentiment_indices = [i for i, sentiment in enumerate(sentiment_result) if sentiment == 'Positif']

        # Proses perubahan: jika prediksi sarkasme adalah "sarcasm", ubah sentiment positif menjadi negatif
        for idx in positive_sentiment_indices:
            if sarcasm_pred[idx] == 'Sarcasm':
                sentiment_result[idx] = 'Negatif'

        # Return hasil sentiment setelah sarkasme
        return sentiment_result
    
    # Gabungkan kedua pipeline untuk sentiment dan sarcasm
class CombinedPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, sentiment_pipeline, sarcasm_pipeline):
        self.sentiment_pipeline = sentiment_pipeline
        self.sarcasm_pipeline = sarcasm_pipeline
        self.sarcasm_checker = SentimentAfterSarcasmTransformer()

    def fit(self, X, y=None):
        # Pipeline hanya memerlukan fit jika Anda ingin melatihnya, untuk prediksi bisa dilewati
        return self

    def predict(self, X):
        # Proses input melalui kedua pipeline
        sentiment_result = self.sentiment_pipeline.predict(X)
        sarcasm_pred = self.sarcasm_pipeline.predict(X)

        # Menggunakan SentimentAfterSarcasmTransformer untuk memperbarui hasil sentiment
        return self.sarcasm_checker.transform([sentiment_result, sarcasm_pred])
