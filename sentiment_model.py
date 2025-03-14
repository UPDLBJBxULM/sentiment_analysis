import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import joblib  # Untuk menyimpan TF-IDF Vectorizer

# Fungsi untuk membaca file CSV
def baca_data_csv(nama_file):
    df = pd.read_csv(nama_file, encoding='latin1')
    if 'label' in df.columns:
        df.rename(columns={'label': 'sentimen'}, inplace=True)
    return df

# Baca data training dan testing
df_train = baca_data_csv('komentar_train.csv')
df_test = baca_data_csv('komentar_test.csv')

# Menghapus baris yang memiliki nilai NaN pada kolom 'komentar' dan 'sentimen'
df_train = df_train.dropna(subset=['komentar', 'sentimen'])
df_test = df_test.dropna(subset=['komentar', 'sentimen'])

# Pastikan hanya label yang valid yang digunakan
valid_labels = ['Positif', 'Netral', 'Negatif']
df_train = df_train[df_train['sentimen'].isin(valid_labels)]
df_test = df_test[df_test['sentimen'].isin(valid_labels)]

# Persiapkan data untuk klasifikasi
def persiapkan_teks(texts):
    return [' '.join(word_tokenize(str(text).lower())) for text in texts]

X_train_text = persiapkan_teks(df_train['komentar'])
X_test_text = persiapkan_teks(df_test['komentar'])

# Menggunakan TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test_text).toarray()

# Simpan TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Encoding label dengan LabelEncoder()
label_encoder = LabelEncoder()
label_encoder.fit(valid_labels)  # Pastikan hanya 3 kelas

y_train_encoded = label_encoder.transform(df_train['sentimen'])
y_test_encoded = label_encoder.transform(df_test['sentimen'])

# Konversi label menjadi One-Hot Encoding
y_train_categorical = to_categorical(y_train_encoded, num_classes=3)
y_test_categorical = to_categorical(y_test_encoded, num_classes=3)

# Membangun Model Keras
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # Output 3 kelas (Positif, Netral, Negatif)
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping untuk mencegah overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Latih Model
model.fit(X_train_tfidf, y_train_categorical, epochs=10, batch_size=16,
          validation_split=0.2, callbacks=[early_stopping])

# Simpan Model ke File .h5
model.save('sentiment_model.h5')
print("Model berhasil disimpan sebagai 'sentiment_model.h5'")

