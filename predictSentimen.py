from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Load model dan alat preprocessing
model = load_model('sentiment_model2.h5')
tfidf_vectorizer = joblib.load('tfidf_vectorizer2.pkl')
label_encoder = joblib.load('label_encoder2.pkl')

# Fungsi untuk memproses teks
def preprocess_texts(texts):
    processed_texts = [' '.join(word_tokenize(text.lower())) for text in texts]
    text_tfidf = tfidf_vectorizer.transform(processed_texts).toarray()
    return text_tfidf

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Log request data
    logging.info(f"Received request: {data}")

    if not data or "data" not in data:
        return jsonify({'error': 'Request harus berisi daftar komentar dalam key "data"'}), 400

    komentar_list = [item.get('komentar', '') for item in data["data"]]
    row_list = [item.get('row', None) for item in data["data"]]

    # Pastikan tidak ada komentar kosong
    if not all(komentar_list):
        return jsonify({'error': 'Semua komentar harus memiliki nilai'}), 400

    # Preprocess input
    input_tfidf = preprocess_texts(komentar_list)

    # Prediksi dengan model
    predictions = model.predict(input_tfidf)
    predicted_labels = np.argmax(predictions, axis=1)  # Ambil indeks dengan probabilitas tertinggi
    sentimen_list = label_encoder.inverse_transform(predicted_labels)  # Ubah ke label aslinya

    # Format response
    result = [{"row": row, "komentar": komentar, "sentimen": sentimen} 
              for row, komentar, sentimen in zip(row_list, komentar_list, sentimen_list)]

    logging.info(f"Response: {result}")

    return jsonify({"data": result})

# Jalankan server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
