import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize

# Fungsi untuk membaca file CSV
def baca_data_csv(nama_file):
    df = pd.read_csv(nama_file, encoding='latin1')
    if 'label' in df.columns:
        df.rename(columns={'label': 'sentimen'}, inplace=True)
    return df

# Fungsi untuk memproses teks
def persiapkan_teks(texts):
    return [' '.join(word_tokenize(str(text).lower())) for text in texts]

# Muat Model dan TF-IDF Vectorizer
model = load_model('sentiment_model.h5')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Data uji baru
df_new = baca_data_csv('data_test_komentar(without).csv')
df_new = df_new.dropna(subset=['komentar'])  # Hapus baris kosong
df_new = df_new[~df_new['komentar'].isin(['-', '.'])]  # Hapus entri tidak valid

# Proses teks baru
X_new_text = persiapkan_teks(df_new['komentar'])
X_new_tfidf = tfidf_vectorizer.transform(X_new_text).toarray()

# Prediksi Sentimen
prediksi = model.predict(X_new_tfidf)

# Konversi hasil prediksi ke kategori sentimen
label_mapping = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
df_new['prediksi_sentimen'] = [label_mapping[np.argmax(p)] for p in prediksi]

# Simpan hasil
df_new.to_csv('hasil_prediksi.csv', index=False)
print("Hasil prediksi telah disimpan dalam 'hasil_prediksi.csv'")
