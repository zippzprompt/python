#import modul flask
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model yang telah dilatih
model = pickle.load(open("gabel_klasifikasi.pkl", "rb"))

# Kontrol halaman utama
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/produk')
def produk():
    return render_template('produk.html')

@app.route('/rekomendasi', methods=["POST", "GET"])
def rekomendasi():
    prediksi = None  # Default nilai prediksi

    if request.method == "POST":
        try:
            # Ambil input dari form HTML
            jumlah_pengguna = int(request.form['Jumlah_Pengguna'])
            pengguna_harian = int(request.form['Pengguna_Harian'])
            aktivitas_utama = int(request.form['Aktivitas_Utama'])
            perangkat_utama = int(request.form['Perangkat_Utama'])
            kecepatan = int(request.form['Kecepatan'])

            # Konversi ke DataFrame untuk prediksi
            input_data = pd.DataFrame([[jumlah_pengguna, pengguna_harian, aktivitas_utama, perangkat_utama, kecepatan]],
                                      columns=['Jumlah_Pengguna', 'Pengguna_Harian', 'Aktivitas_Utama', 'Perangkat_Utama', 'Kecepatan'])

            # Lakukan prediksi
            predictions = model.predict(input_data)

            # Pastikan prediksi adalah string yang valid
            rekomendasi = ["A", "B", "C", "D"]
            if predictions[0] in rekomendasi:
                prediksi = predictions[0]  # Ambil elemen pertama dari hasil prediksi
        except Exception as e:
            print(f"Terjadi kesalahan: {e}")  # Debugging error

    # Render halaman dengan hasil prediksi
    return render_template('rekomendasi.html', prediksi=prediksi)

if __name__ == '__main__':
    app.run(debug=True)
