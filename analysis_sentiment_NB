import nltk
nltk.download('punkt_tab')

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

# Persiapkan data untuk klasifikasi
def persiapkan_teks(texts):
    return [' '.join(word_tokenize(str(text).lower())) for text in texts]

# Proses teks untuk model
X_train_text = persiapkan_teks(df_train['komentar'])
X_test_text = persiapkan_teks(df_test['komentar'])

y_train = df_train['sentimen']
y_test = df_test['sentimen']

# Buat pipeline untuk klasifikasi
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=0.1))
])

# Latih model
sentiment_pipeline.fit(X_train_text, y_train)

# Evaluasi model
y_pred = sentiment_pipeline.predict(X_test_text)
print("\n=== Evaluasi Model ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sentiment_pipeline.classes_, yticklabels=sentiment_pipeline.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Membaca dan membersihkan data baru
df_new = baca_data_csv('Data uji.csv')
df_new = df_new.dropna(subset=['komentar'])  # Hapus baris dengan NaN
df_new = df_new[~df_new['komentar'].isin(['-', '.'])]  # Hapus data dengan nilai '-' atau '.'

# Pastikan hanya baris dengan komentar valid yang diproses
valid_comments = df_new['komentar'].notna() & (df_new['komentar'].str.strip() != '')
df_new = df_new[valid_comments]

X_new_text = persiapkan_teks(df_new['komentar'])

df_new['prediksi_sentimen'] = sentiment_pipeline.predict(X_new_text)

# Hapus kolom 'Unnamed: 1' jika ada
if 'Unnamed: 1' in df_new.columns:
    df_new = df_new.drop(columns=['Unnamed: 1'])

# Simpan hasil prediksi ke CSV tanpa menyertakan index
df_new.to_csv('hasil_prediksi.csv', index=False)
print("Hasil prediksi telah disimpan dalam 'hasil_prediksi.csv'")

print(df_new['prediksi_sentimen'].value_counts())
print(df_new[['komentar', 'prediksi_sentimen']].head(10))
if df_new.empty:
    print("Data hasil prediksi kosong!")

def buat_wordcloud(sentimen, warna):
    df_filtered = df_new[df_new['prediksi_sentimen'].str.lower() == sentimen.lower()]
    
    if df_filtered.empty:
        print(f"Tidak ada data untuk sentimen {sentimen}, WordCloud tidak dapat dibuat.")
        return
    
    teks = ' '.join(df_filtered['komentar'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap=warna, random_state=42).generate(teks)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud untuk Sentimen {sentimen}')
    plt.show()

buat_wordcloud('Positif', 'coolwarm')
buat_wordcloud('Negatif', 'magma')
buat_wordcloud('Netral', 'viridis')

