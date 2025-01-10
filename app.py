from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import pandas as pd
import joblib
import openpyxl
from io import BytesIO

from models.text_preprocessor import TextPreprocessor, SentimentAfterSarcasmTransformer, CombinedPipeline

# Load pre-trained models
sentiment = joblib.load('models/pipeline_sentiment.pkl')
sarcasm = joblib.load('models/pipeline_sarcasm.pkl')
sentiment_after_sarcasm = joblib.load('models/pipeline_sentiment_after_sarcasm.pkl')

app = Flask(__name__)

analyzed_data = None
processed_file = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    if not data or 'textInput' not in data:
        return jsonify({"error": "'textInput' field is missing"}), 400

    text = data['textInput']

    try:
        # Predictions
        sarcasm_prediction = sarcasm.predict([text])[0]
        sentiment_sebelum = sentiment.predict([text])[0]
        sentiment_prediction = sentiment_after_sarcasm.predict([text])[0]

        return jsonify({
            "sentiment_sebelum": sentiment_sebelum,
            "sentimen": sentiment_prediction,
            "sarcasm": sarcasm_prediction,
            "text": text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/upload_file', methods=['POST'])
def upload_csv_or_xlsx():
    global analyzed_data, processed_file
    file = request.files.get('File')

    if not file:
        return "No file uploaded", 400

    try:
        filename = file.filename.lower()

        if filename.endswith('.csv'):
            df = pd.read_csv(file, delimiter=',', quotechar='"', skipinitialspace=True)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return "Unsupported file format. Please upload a CSV or XLSX file.", 400

        if 'textDisplay' not in df.columns:
            return "'textDisplay' column not found in uploaded file", 400

        # Process the 'textDisplay' data kolom
        df['textDisplay'] = df['textDisplay'].fillna('').astype(str)
        df['sentiment_before'] = sentiment.predict(df['textDisplay'])
        df['sarcasm_prediction'] = sarcasm.predict(df['textDisplay'])[0]
        df['sentiment_result'] = sentiment_after_sarcasm.predict(df['textDisplay'])


        # Store the processed data globally
        analyzed_data = df
        processed_file = BytesIO()
        df.to_csv(processed_file, index=False)
        processed_file.seek(0)

        return redirect(url_for('show_results'))

    except Exception as e:
        return str(e), 500

@app.route('/results')
def show_results():
    global analyzed_data
    if analyzed_data is None:
        return redirect(url_for('index'))
    
    table_html = analyzed_data.to_html(classes='table table-striped table-bordered', border=0, index=False)
    table_html = table_html.strip()

    return render_template('results.html', tables=table_html)

@app.route('/download_processed_file')
def download_processed_file():
    global processed_file
    if processed_file is None:
        return redirect(url_for('index'))
    return send_file(processed_file, as_attachment=True, download_name='processed_sentiments.csv', mimetype='text/csv')

if __name__ == '__main__':
    app.run(debug=True)