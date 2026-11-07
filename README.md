# Deteksi Akun Bot dan Buzzer

Proyek penelitian berbasis Python untuk menganalisis apakah akun media sosial menunjukkan sinyal statistik yang berkaitan dengan otomatisasi, perilaku bot, spam, atau koordinasi. Sistem menggabungkan fitur profil akun, karakteristik teks, serta pola koordinasi waktu dan konten.

> **Peringatan penting:** keluaran model adalah sinyal probabilistik, bukan bukti bahwa seseorang atau suatu akun adalah bot, buzzer, spammer, atau memiliki niat tertentu. Hasil wajib ditinjau manusia dan tidak boleh digunakan sebagai satu-satunya dasar untuk menuduh, menghukum, atau menargetkan akun.

## Tujuan proyek

Proyek ini digunakan untuk:

- membangun baseline machine learning untuk deteksi akun bot;
- membandingkan beberapa kelompok fitur: profil, teks, dan koordinasi;
- menguji performa model pada satu dataset maupun lintas dataset;
- menghasilkan skor probabilitas untuk data akun baru;
- mengeksplorasi data publik terkait aktivitas akun dan tweet.

Proyek ini **bukan** alat untuk memastikan identitas, afiliasi, niat, atau status hukum pemilik akun.

## Teknologi

- Python 3
- pandas dan NumPy untuk pemrosesan data
- scikit-learn untuk fitur, pipeline, model, dan evaluasi
- SciPy untuk struktur data numerik sparse
- Matplotlib untuk grafik precision-recall
- XGBoost untuk model tambahan jika tersedia
- pytest untuk pengujian fitur
- Jupyter Notebook untuk eksplorasi dataset tertentu

## Struktur proyek

```text
.
├── bot-detector/
│   ├── src/
│   │   ├── features.py
│   │   ├── pipeline.py
│   │   ├── predict_account.py
│   │   ├── analyze_indonesia.py
│   │   ├── inspect_data.py
│   │   ├── inspect_support.py
│   │   ├── make_synthetic.py
│   │   ├── smoke_test.py
│   │   └── train_baseline.py
│   ├── tests/
│   │   └── test_features.py
│   ├── results/
│   ├── requirements.txt
│   └── README.md
├── data/
│   ├── akun_uji.csv
│   ├── dataset1/
│   ├── dataset2/
│   ├── data_bot3/
│   └── data_bot4/Twibot-20/Twibot-20/
├── requirements.txt
└── .gitignore
```

### Penjelasan file utama

#### `bot-detector/src/features.py`

Menyediakan fitur yang dapat digunakan kembali untuk:

- atribut akun: followers, following, rasio followers/following, jumlah posting, umur akun, verifikasi, dan bio;
- atribut konten: panjang teks, jumlah kata, URL, mention, hashtag, dan retweet;
- koordinasi: duplikasi teks, hashtag yang sama, target retweet yang sama, dan aktivitas pada jendela waktu 10 menit.

#### `bot-detector/src/pipeline.py`

Pipeline utama untuk membaca data, mendeteksi nama kolom, membangun fitur, melatih model, dan mengevaluasi hasil. Pipeline mendukung file CSV, TSV, Excel, JSON, JSON Lines, serta folder berisi beberapa file data yang dapat digabung berdasarkan kolom ID.

Model yang tersedia:

- Logistic Regression;
- Random Forest;
- XGBoost jika paketnya terpasang.

#### `bot-detector/src/predict_account.py`

Melatih model referensi dari dataset berlabel, lalu memberi skor pada file akun baru yang tidak memiliki label. Hasil mencakup probabilitas bot dan tingkat sinyal `rendah`, `sedang`, atau `tinggi`.

#### `bot-detector/tests/test_features.py`

Berisi pengujian untuk memastikan fitur koordinasi menghitung teks duplikat dan tetap menghasilkan kolom yang benar ketika timestamp tidak tersedia.

#### `data/data_bot3/`

Dataset tweet komunitas techbro Indonesia beserta data aktivitas, topik, contoh kode, dan notebook eksplorasi. Dataset ini ditujukan untuk eksplorasi NLP dan sosiolinguistik, bukan label bot otomatis.

#### `data/dataset1/`, `data/dataset2/`, dan `data/data_bot4/`

Kumpulan dataset berlabel atau dataset pendukung untuk eksperimen deteksi bot, termasuk dataset TwiBot-20 dan dataset Twitter Bot Detection yang diperoleh melalui Kaggle. Struktur file lokal dapat berbeda dari struktur file asli di Kaggle karena proses pengunduhan, ekstraksi, atau pengolahan ulang.

## Instalasi

Masuk ke folder modul utama:

```bash
cd bot-detector
```

Buat virtual environment baru jika belum tersedia:

```bash
python -m venv .venv
```

Aktifkan environment di Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Aktifkan environment di Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Instal dependensi:

```bash
python -m pip install -r requirements.txt
```

Dependensi opsional:

```bash
python -m pip install shap pytest
```

`torch`, `transformers`, dan `sentencepiece` hanya diperlukan jika pengembangan berikutnya menambahkan model IndoBERT. Pipeline baseline saat ini menggunakan TF-IDF dan tidak membutuhkan IndoBERT.

## Menjalankan pipeline evaluasi

Semua perintah berikut dijalankan dari folder `bot-detector`.

### Satu dataset CSV

```bash
python src/pipeline.py ../data/dataset1/bot_detection_data.csv
```

### Dataset JSON TwiBot-20

```bash
python src/pipeline.py ../data/dataset1/train.json
```

### Membandingkan beberapa dataset

```bash
python src/pipeline.py ../data/dataset1/bot_detection_data.csv ../data/dataset2
```

Satu argumen dapat berupa file atau folder. Jika folder berisi beberapa file data dengan kolom ID yang cocok, file akan digabungkan berdasarkan ID akun.

## Format data yang didukung

Pipeline membaca:

- CSV;
- TSV;
- Excel `.xlsx` atau `.xls`;
- JSON berbentuk list objek;
- JSON berbentuk `{ "data": [...] }` atau struktur list serupa;
- JSON berbentuk objek dengan ID sebagai key;
- JSON Lines `.jsonl` dan `.ndjson`;
- folder yang berisi beberapa file data.

Pipeline mencoba mengenali nama kolom umum secara otomatis. Contoh alias yang didukung:

| Informasi | Contoh nama kolom |
|---|---|
| Label | `label`, `bot`, `is_bot`, `class`, `target`, `bot_label` |
| ID akun | `id`, `user_id`, `id_str`, `account_id` |
| Followers | `followers_count`, `followers`, `follower_count` |
| Following | `friends_count`, `following`, `following_count` |
| Jumlah posting | `statuses_count`, `tweet_count`, `tweets`, `posts` |
| Verifikasi | `verified`, `is_verified` |
| Tanggal | `created_at`, `account_creation`, `creation_date` |
| Nama akun | `screen_name`, `username`, `user_name`, `handle` |
| Bio | `description`, `bio`, `profile_description` |
| Teks | `text`, `tweet`, `tweets_text`, `content` |

Dataset untuk evaluasi harus memiliki kolom label. Label numerik `0` dan `1` digunakan langsung; label seperti `bot`, `automated`, `spam`, atau `fake` dipetakan menjadi bot, sedangkan label lain diperlakukan sebagai non-bot sesuai aturan pipeline.

## Fitur yang dibuat

### Fitur profil

- jumlah followers dan following;
- rasio followers terhadap following;
- jumlah posting, favorit, dan akun yang mencantumkan akun tersebut;
- transformasi log untuk nilai jumlah yang besar;
- umur akun;
- posting per hari dan followers per hari;
- status verifikasi;
- penggunaan profil atau gambar default;
- keberadaan URL profil;
- panjang nama pengguna dan jumlah digit;
- panjang, URL, hashtag, mention, serta kekosongan bio.

### Fitur teks

Teks bio dan tweet dapat digabungkan lalu diproses menggunakan TF-IDF unigram dan bigram. URL dinormalisasi menjadi token `URL` sebelum proses TF-IDF.

### Fitur koordinasi

Fitur koordinasi dihitung dari keseluruhan observasi yang tersedia, antara lain:

- jumlah baris lain dengan teks yang sama;
- jumlah akun atau baris lain dengan hashtag yang sama;
- jumlah baris lain yang me-retweet target yang sama;
- jumlah aktivitas pada jendela waktu 10 menit.

Fitur-fitur tersebut hanya menunjukkan pola kemiripan atau waktu yang sama. Kemiripan tidak otomatis berarti koordinasi yang disengaja.

## Model dan evaluasi

Pipeline menguji tiga konfigurasi fitur:

1. `profil`;
2. `profil+teks`;
3. `profil+teks+koordinasi`.

Untuk setiap konfigurasi, pipeline menjalankan:

- evaluasi dalam dataset dengan stratified train-test split;
- evaluasi lintas dataset untuk melihat apakah model dapat berpindah ke dataset lain.

Metrik yang disimpan:

- PR-AUC;
- ROC-AUC;
- F1-score pada threshold 0,5.

Evaluasi lintas dataset lebih relevan untuk menilai generalisasi, tetapi tetap tidak membuktikan bahwa model berlaku untuk semua platform, negara, bahasa, atau periode waktu.

## Output

Hasil pipeline disimpan di `bot-detector/results/`:

- `results.csv`: ringkasan metrik setiap model dan skenario;
- `column_mapping.txt`: pemetaan kolom input yang berhasil dikenali;
- `pr_curve_*.png`: kurva precision-recall;
- `shap_summary.png`: ringkasan kontribusi fitur jika `shap` dan `xgboost` tersedia.

Periksa `column_mapping.txt` setiap kali menggunakan dataset baru. Kolom yang salah dikenali atau tidak terbaca dapat menyebabkan hasil evaluasi menyesatkan.

## Prediksi akun baru

File target tidak perlu memiliki kolom label. Contoh:

```bash
python src/predict_account.py ../data/akun_uji.csv
```

Menentukan dataset referensi dan lokasi output secara eksplisit:

```bash
python src/predict_account.py ../data/akun_uji.csv \
  --reference ../data/dataset1/train.json \
  --out results/account_predictions.csv
```

Kolom output:

- `username` atau `row`: identitas baris input;
- `bot_probability`: probabilitas keluaran model;
- `signal_level`: kategori `rendah`, `sedang`, atau `tinggi`.

Probabilitas tersebut bergantung pada dataset referensi, fitur yang tersedia, kualitas label, dan distribusi data. Nilainya tidak boleh dianggap sebagai persentase kepastian bahwa akun adalah bot.

## Menjalankan pengujian

Dari folder `bot-detector`:

```bash
python -m pytest tests
```

Jika pytest tidak tersedia:

```bash
python -m pip install pytest
```

## Reproduksibilitas dan praktik penggunaan

- Gunakan data berlabel yang jelas dan dokumentasikan sumbernya.
- Pisahkan akun yang sama antara data latih dan data uji agar tidak terjadi kebocoran data.
- Validasi model pada bahasa, platform, dan periode yang sesuai dengan tujuan analisis.
- Audit ketidakseimbangan kelas dan perubahan distribusi data.
- Jangan menghapus data mentah secara permanen sebelum hasil dapat direproduksi.
- Jangan menyimpan token API, kredensial, atau data pribadi ke repository.
- Gunakan hanya data publik dan patuhi ketentuan layanan platform serta aturan yang berlaku.
- Simpan keputusan akhir pada proses tinjauan manusia, bukan pada skor model semata.

## Keterbatasan

- Dataset publik dapat memiliki bias pelabelan dan bias pengambilan sampel.
- Pola bot berubah dari waktu ke waktu sehingga performa model dapat menurun.
- Akun manusia juga dapat memiliki frekuensi posting, teks duplikat, atau pola waktu yang mirip bot.
- Data TwiBot-20 bukan dataset khusus bahasa Indonesia.
- Dataset komunitas techbro Indonesia tidak otomatis merupakan dataset bot berlabel.
- Model baseline belum menggunakan IndoBERT atau model bahasa khusus Indonesia.
- Fitur koordinasi yang dihitung dari satu kumpulan data tidak membuktikan hubungan antar-akun.

## Sumber dataset dan referensi Kaggle

Dataset dan referensi utama yang digunakan dalam proyek ini berasal dari Kaggle:

1. [TwiBot-20](https://www.kaggle.com/datasets/marvinvanbo/twibot-20) — dataset untuk penelitian deteksi bot Twitter.
2. [Twitter Bot Detection Dataset](https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset) — dataset profil dan tweet dengan label deteksi bot.
3. [TwiBot20 Data](https://www.kaggle.com/datasets/siyareac/twibot20data) — sumber alternatif atau salinan dataset TwiBot-20 untuk eksperimen.
4. [Knowledge Graph NLP Tutorial: BERT, spaCy, NLTK](https://www.kaggle.com/code/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk) — notebook referensi teknik NLP, bukan dataset utama proyek.

Cara mengunduh dataset dari Kaggle:

1. Buka halaman dataset dan setujui ketentuan penggunaan jika diminta.
2. Unduh file ZIP dari Kaggle.
3. Ekstrak file ke folder `data/`.
4. Pastikan file hasil ekstraksi memiliki format yang didukung pipeline.
5. Jalankan pipeline menggunakan path file atau folder dataset tersebut.

Contoh setelah dataset diekstrak:

```bash
python src/pipeline.py ../data/dataset1/train.json
```

Periksa file `README`, `README.txt`, atau `LICENSE` di masing-masing folder dataset untuk mengetahui lisensi dan kewajiban atribusi. Jangan menganggap seluruh dataset di dalam repository memiliki lisensi yang sama. Ikuti lisensi, aturan Kaggle, dan ketentuan penggunaan dari pemilik dataset.

## Catatan pengembangan

README ini mendokumentasikan kondisi proyek berdasarkan file yang tersedia saat ini. Jika menambahkan model, sumber data, format kolom, atau langkah preprocessing baru, perbarui bagian struktur proyek, instalasi, penggunaan, dan keterbatasan secara bersamaan. 
