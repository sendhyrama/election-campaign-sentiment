from flask import Flask, render_template, request, jsonify
import os
import joblib

from models.text_preprocessor import TextPreprocessor,SentimentAfterSarcasmTransformer,CombinedPipeline

# Load pre-trained models
sentiment = joblib.load('models/pipeline_sentiment.pkl')
sarcasm = joblib.load('models/pipeline_sarcasm.pkl')
sentiment_after_sarcasm = joblib.load('models/pipeline_sentiment_after_sarcasm.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    if request.method == 'POST':
        # Get the text input from the form
        text = request.form.get('textInput')  # Using .get() to avoid KeyError

        # Check if the text input is present
        if not text:
            return jsonify({"error": "'textInput' field is missing"}), 400

        # Process sentiment and sarcasm
        sarcasm_prediction = sarcasm.predict([text])[0]  # Predict sarcasm
        sentiment_sebelum = sentiment.predict([text])[0]  # Predict sarcasm
        sentiment_prediction = sentiment_after_sarcasm.fit(text, y=None).predict([text])[0]  # Sentiment after sarcasm

        # Render the results in the coba.html page
        return render_template('coba.html', 
                               sentimen=sentiment_prediction,
                               sentiment_sebelum = sentiment_sebelum,
                               sarcasm=sarcasm_prediction,
                               text=text)


if __name__ == '__main__':
    app.run(debug=True)
