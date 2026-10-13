# Fraud-LSTM: Deteksi Anomali Transaksi dengan LSTM-Autoencoder

Repositori ini membandingkan tiga pendekatan deteksi anomali tak terawasi (*unsupervised*) pada data transaksi: LSTM-Autoencoder, Standard Autoencoder, dan Isolation Forest. Modelnya dilatih hanya pada transaksi normal, lalu transaksi dengan galat rekonstruksi tinggi ditandai sebagai anomali.

Proyek ini penelitian skala kecil dan hasilnya belum menunjukkan sistem yang siap dipakai. Hasil pada data fraud nyata masih rendah (lihat bagian Hasil dan Keterbatasan).

## Data

| Dataset | Isi | Catatan |
|---|---|---|
| Kaggle Credit Card Fraud | 284.807 transaksi kartu kredit, fraud 0,17% | Label fraud asli. Data uji berisi 42.720 transaksi dengan 52 fraud (0,12%). |
| Olist (e-commerce Brasil) | Pesanan, pembayaran, dan pelanggan | **Tidak punya label fraud.** Anomali dibuat sendiri dengan injeksi sintetis, sehingga ini simulasi, bukan fraud nyata. |

Berkas dataset tidak ikut di repositori karena ukurannya besar. Folder `Data/` hanya berisi penunjuk Git LFS. Unduh dataset dari Kaggle ("Credit Card Fraud Detection" oleh ULB dan "Brazilian E-Commerce Public Dataset by Olist"), lalu letakkan di `Data/Kaggle_Fraud/creditcard.csv` dan `Data/Olist/`. Jalur di beberapa notebook awal masih jalur lokal Windows dan perlu disesuaikan.

## Cara kerja

1. **Fitur.** Kaggle memakai V1 sampai V28, Amount (distandarkan), dan jam dalam bentuk sin dan cos. Olist memakai `payment_installments`, `freight_value`, jam, dan hari dalam seminggu (sin dan cos).
2. **Sekuens.** Jendela `TIMESTEP = 2`. Di Olist, jendela dibentuk per pelanggan. Di Kaggle tidak ada ID pelanggan, jadi jendela berisi dua transaksi yang berurutan menurut waktu, bukan riwayat satu pengguna.
3. **Pembagian data dan pelatihan.** Kaggle dibagi menurut urutan waktu (70% latih, 15% validasi, 15% uji). Data latih hanya berisi transaksi normal. Di Olist, scaler dilatih hanya pada data latih normal. Di Kaggle, scaler dilatih pada seluruh data sebelum pembagian (lihat Keterbatasan).
4. **Model.** LSTM-Autoencoder dengan dua lapis LSTM di encoder (64 dan 32 unit), ruang laten 16, dropout 0,2, Adam (learning rate 0,001), MSE, batch 256, maksimum 100 epoch dengan *early stopping* (patience 10). Pembanding: Autoencoder biasa pada data yang diratakan, dan Isolation Forest.
5. **Threshold.** Kaggle memakai persentil ke-99,5 dari galat rekonstruksi transaksi normal di validation set. Olist memakai threshold dengan F1 terbaik pada validation set.
6. **Injeksi anomali Olist.** Ada dua tipe: statis (`freight_value` dikali 5) dan "temporal" (`payment_installments` diubah dari 1 menjadi 3 pada transaksi yang didahului transaksi 1 cicilan). Jumlah tiap tipe di data uji berbeda (226 statis, 398 temporal).

## Hasil

Semua angka berasal dari satu kali pelatihan dengan satu seed pada data uji, tanpa uji statistik. Sumbernya `Notebooks/05_Evaluation_and_Results.ipynb` dan `Notebooks/outputs/FINAL_EVALUATION_REPORT.txt`.

### Kaggle (52 fraud dari 42.720 transaksi uji)

| Model | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| LSTM-AE | 0,0278 | 0,0577 | 0,0375 | 3 | 105 | 49 |
| Standard AE | 0,1704 | 0,4423 | 0,2460 | 23 | 112 | 29 |
| Isolation Forest | 0,0000 | 0,0000 | 0,0000 | 0 | 30 | 52 |

Ketiga model tidak memadai untuk data ini. Standard AE terbaik, dan LSTM-AE justru hampir tidak mendeteksi fraud.

### Olist dengan anomali injeksi (624 anomali dari 3.205 sampel uji)

| Model | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| LSTM-AE | 0,4017 | 0,4455 | 0,4225 | 278 | 414 | 346 |
| Standard AE | 0,3250 | 0,3317 | 0,3283 | 207 | 430 | 417 |
| Isolation Forest | 0,2106 | 0,2933 | 0,2451 | 183 | 686 | 441 |

LSTM-AE punya F1 tertinggi, 28,7% lebih tinggi secara relatif daripada Standard AE. Selisih ini belum diuji secara statistik.

### Berdasarkan tipe anomali (F1, Olist)

| Tipe | Sampel | LSTM-AE | Standard AE | Isolation Forest |
|---|---|---|---|---|
| Statis (freight x5) | 226 | 0,9704 | 0,8528 | 0,7011 |
| Temporal (cicilan 1 menjadi 3) | 398 | 0,2808 | 0,1785 | 0,2658 |

Pada anomali temporal, F1 LSTM-AE masih rendah. Hipotesis bahwa model sekuensial menangkap pola antar-transaksi hanya didukung sebagian.

## Keterbatasan

- Anomali Olist dibuat dengan injeksi, jadi hasilnya tidak bisa dibaca sebagai kinerja pada fraud nyata.
- Anomali "temporal" mengubah satu nilai fitur (`payment_installments`), sehingga model statis juga bisa melihatnya. Ini bukan pola sekuensial murni.
- Scaler Kaggle (termasuk Amount) dilatih pada seluruh data sebelum pembagian waktu, sehingga statistik data uji ikut masuk ke penskalaan. Pengaruhnya kemungkinan kecil, tetapi seharusnya scaler hanya dilatih pada data latih.
- Hanya 52 fraud di data uji Kaggle, dan hasil berasal dari satu seed. Belum ada validasi silang, interval kepercayaan, atau uji signifikansi.
- Di Kaggle, "sekuens" hanyalah dua transaksi berurutan menurut waktu karena tidak ada ID pelanggan.
- Notebook `06_Helper_Functions_and_Analysis.ipynb` berisi kelas `RealTimeFraudDetector` yang mensimulasikan inferensi per transaksi. Itu simulasi, belum ada layanan atau API yang dideploy.
- Berkas dataset dan beberapa keluaran (`.csv`, `.npy` di `Notebooks/outputs/data/`) disimpan lewat Git LFS dan tidak tersedia tanpa `git lfs pull`.

## Struktur

```
Notebooks/
  01_Data_Loading_and_EDA.ipynb
  02_Preprocessing_and_Feature_Engineering.ipynb   # fitur, jendela, injeksi anomali Olist
  03_LSTM_Autoencoder_Training.ipynb
  04_Baseline_Models_Training.ipynb                # Standard AE dan Isolation Forest
  05_Evaluation_and_Results.ipynb                  # tabel dan laporan hasil
  06_Helper_Functions_and_Analysis.ipynb           # fungsi bantu dan simulasi inferensi
  Data_Simulation*.ipynb, Training_*.ipynb,
  Preprocessing_Olist.ipynb, Fraud_Done.ipynb      # percobaan awal, sebelum notebook 01-06
  outputs/                                         # model, scaler, gambar, laporan
Output/                                            # keluaran percobaan awal
```

Jalankan notebook 01 sampai 06 secara berurutan. Percobaan awal dipertahankan sebagai riwayat dan tidak dipakai untuk hasil di atas.

## Lingkungan

Python 3, PyTorch, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, joblib. GPU opsional (notebook memakai CUDA jika tersedia).
