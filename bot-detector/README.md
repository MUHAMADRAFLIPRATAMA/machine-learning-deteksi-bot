# Deteksi Bot dan Buzzer Terkoordinasi

Sistem penelitian untuk menilai apakah akun menunjukkan pola otomatisasi atau koordinasi. Skor bukan bukti identitas/niat dan tidak boleh dipakai untuk menghukum akun tanpa tinjauan manusia.

## Fitur dan pendekatan

1. **Akun**: followers, following, rasio, umur akun, posting per hari, verifikasi, bio kosong/panjang.
2. **Konten**: panjang teks, jumlah URL/mention/hashtag, retweet, TF-IDF n-gram.
3. **Jaringan/waktu**: copy-paste, co-hashtag, co-retweet, dan jumlah posting pada jendela 10 menit.
4. **Model**: Logistic Regression, Random Forest, dan XGBoost jika terpasang.
5. **Metrik**: PR-AUC, ROC-AUC, F1, dan kurva precision-recall.

TF-IDF dipakai sebagai baseline teks yang murah dan dapat direproduksi. IndoBERT dapat ditambahkan kemudian dengan `transformers` dan model bahasa Indonesia; tidak wajib untuk menjalankan pipeline dasar.

## Instalasi

Gunakan virtual environment yang sudah tersedia:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Menjalankan

Dari folder `bot-detector`:

```bash
.venv/Scripts/python.exe src/pipeline.py ../data/dataset1/bot_detection_data.csv
```

TwiBot-20:

```bash
.venv/Scripts/python.exe src/pipeline.py ../data/dataset1/train.json
```

Output ada di `results/`: `results.csv`, `column_mapping.txt`, grafik PR, dan SHAP jika tersedia.

## Dataset

- `bot_detection_data.csv`: dataset berlabel bot dengan fitur profil/konten terbatas.
- `TwiBot-20`: dataset publik untuk riset deteksi bot. Bukan dataset Indonesia khusus.
- Data sintetis, jika digunakan, harus diberi label **SIMULASI** dan hanya menguji apakah pipeline mampu menangkap pola yang sengaja dibuat. Simulasi bukan bukti perilaku akun nyata.

## Interpretasi dan etika

PR-AUC sekitar 0,5 pada data seimbang berarti performa mendekati tebakan acak. Hindari klaim generalisasi Indonesia jika belum ada validasi berbahasa Indonesia. Split data harus mencegah akun/tweet yang sama masuk train dan test. Gunakan threshold setelah validasi, audit bias, dan pemeriksaan manusia. Jangan mengekspos atau menargetkan individu berdasarkan skor model.
