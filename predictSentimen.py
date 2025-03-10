from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import logging
import re  # Add this import for regex functions
import string  # Add this for string.punctuation
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Load model dan alat preprocessing
model = load_model('sentiment_model7.h5')
tokenizer = joblib.load('tokenizer.pkl')  # Menggunakan tokenizer.pkl
label_encoder = joblib.load('label_encoder7.pkl')

# Konfigurasi untuk tokenizer
MAX_SEQUENCE_LENGTH = 50  # Sesuaikan dengan nilai yang digunakan saat pelatihan

# URL Service BigQuery
BIGQUERY_SERVICE_URL = "http://127.0.0.1:5009/"

# Fungsi pembersihan teks
def clean_text(text):
    if isinstance(text, str):
        # Mengubah ke huruf kecil
        text = text.lower()
        # Menghapus tanda baca
        text = ''.join([char for char in text if char not in string.punctuation])
        # Menghapus angka
        text = re.sub(r'\d+', '', text)
        # Menghapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Fungsi untuk memproses teks
def preprocess_texts(texts):
    # Menggunakan tokenizer untuk mengubah teks menjadi sequence
    sequences = tokenizer.texts_to_sequences(texts)
    # Padding sequence agar semua memiliki panjang yang sama
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')  # Add padding='post'
    return padded_sequences

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    logging.info(f"Received request: {data}")

    if not data or "data" not in data:
        return jsonify({'error': 'Request harus berisi daftar komentar dalam key "data"'}), 400

    # Get the original comments and rows
    original_komentar_list = [item.get('komentar', '') for item in data["data"]]
    row_list = [item.get('row', None) for item in data["data"]]

    if not all(original_komentar_list):
        return jsonify({'error': 'Semua komentar harus memiliki nilai'}), 400

    # Clean the comments
    cleaned_komentar_list = [clean_text(komentar) for komentar in original_komentar_list]
    logging.info(f"Original comments: {original_komentar_list}")
    logging.info(f"Cleaned comments: {cleaned_komentar_list}")
    
    # Preprocess input
    processed_input = preprocess_texts(cleaned_komentar_list)

    # Prediksi dengan model
    predictions = model.predict(processed_input)
    predicted_labels = np.argmax(predictions, axis=1)
    sentimen_list = label_encoder.inverse_transform(predicted_labels)

    # Format hasil - use original comments in the result
    result = [{"row": row, "komentar": orig_komentar, "sentimen": sentimen} 
              for row, orig_komentar, sentimen in zip(row_list, original_komentar_list, sentimen_list)]

    logging.info(f"Prediction Result: {result}")

    # Kirim ke Service BigQuery
    try:
        response = requests.post(BIGQUERY_SERVICE_URL, json={"data": result})
        logging.info(f"BigQuery Service Response: {response.json()}")
    except Exception as e:
        logging.error(f"Failed to send data to BigQuery Service: {str(e)}")

    return jsonify({"data": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
