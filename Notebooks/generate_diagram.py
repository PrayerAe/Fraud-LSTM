import os
import warnings

# --- PERBAIKAN KRUSIAL (LANGKAH 1) ---
# Tambahkan path Graphviz Anda (dari 'dot -V') secara manual SEBELUM mengimpor tensorflow
# Gunakan 'r' (raw string) untuk menangani backslashs di Windows
# Ini adalah path yang Anda berikan dari output 'dot -V'
graphviz_bin_path = r"C:\Program Files\Graphviz-14.0.2-win64\bin"

# Cek apakah path itu ada sebelum menambahkannya
if not os.path.isdir(graphviz_bin_path):
    print(f"--- PERINGATAN ---")
    print(f"Folder Graphviz TIDAK ditemukan di: {graphviz_bin_path}")
    print("Pastikan path tersebut benar.")
else:
    print(f"Menambahkan path Graphviz secara manual: {graphviz_bin_path}")
    os.environ["PATH"] += os.pathsep + graphviz_bin_path
# --- AKHIR PERBAIKAN ---


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.utils import plot_model

# Menonaktifkan peringatan TensorFlow yang tidak relevan
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. DEFINISI KONFIGURASI (Sesuai Notebook 2 & 3 Anda) ---
TIMESTEP = 2
N_FEATURES_FRAUD = 31  
N_FEATURES_OLIST = 6   
HIDDEN_DIM1 = 64
HIDDEN_DIM2 = 32
LATENT_DIM = 16

def create_lstm_ae_model(n_features, timestep, hidden_dim1, hidden_dim2, latent_dim, model_name):
    inputs = Input(shape=(timestep, n_features), name="Input_Sequence")
    encoder = LSTM(hidden_dim1, activation='relu', return_sequences=True, name="Encoder_LSTM_1")(inputs)
    encoder = LSTM(hidden_dim2, activation='relu', return_sequences=False, name="Encoder_LSTM_2")(encoder)
    latent_vector = Dense(latent_dim, activation='relu', name="Latent_Vector_Bottleneck")(encoder)
    decoder = Dense(hidden_dim2, activation='relu', name="Decoder_Dense")(latent_vector)
    decoder = RepeatVector(timestep, name="Repeat_Vector")(decoder)
    decoder = LSTM(hidden_dim2, activation='relu', return_sequences=True, name="Decoder_LSTM_1")(decoder)
    decoder = LSTM(hidden_dim1, activation='relu', return_sequences=True, name="Decoder_LSTM_2")(decoder)
    output = TimeDistributed(Dense(n_features, activation='linear'), name="Output_Reconstructed")(decoder)
    model = Model(inputs=inputs, outputs=output, name=model_name)
    return model

print("Membuat arsitektur model...")

# --- 2. MEMBUAT MODEL UNTUK SETIAP DATASET ---
model_fraud = create_lstm_ae_model(
    N_FEATURES_FRAUD, TIMESTEP, 
    HIDDEN_DIM1, HIDDEN_DIM2, LATENT_DIM, 
    "Fraud_Kaggle_Model"
)
model_olist = create_lstm_ae_model(
    N_FEATURES_OLIST, TIMESTEP, 
    HIDDEN_DIM1, HIDDEN_DIM2, LATENT_DIM, 
    "Olist_Simulation_Model"
)

# Tidak perlu print summary, ini hanya membuang waktu
# print("Model Fraud (Kaggle) Summary:")
# model_fraud.summary()
# print("\nModel Olist (Simulasi) Summary:")
# model_olist.summary()

# --- 3. GENERATE GAMBAR DIAGRAM (DENGAN VERIFIKASI SEBENARNYA) ---
output_path = "outputs/figures"
os.makedirs(output_path, exist_ok=True) # Pastikan folder ada

file_fraud = os.path.join(output_path, "FIG2_A_Fraud_Architecture.png")
file_olist = os.path.join(output_path, "FIG2_B_Olist_Architecture.png")

# --- Percobaan Generate Diagram Fraud ---
print(f"\nMencoba men-generate diagram Fraud ke: {file_fraud} ...")
try:
    plot_model(
        model_fraud,
        to_file=file_fraud,
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        rankdir='LR'
    )
    
    # --- PERBAIKAN VERIFIKASI ---
    # Cek apakah file BENAR-BENAR dibuat
    if os.path.exists(file_fraud) and os.path.getsize(file_fraud) > 0:
        print(f"BERHASIL: Diagram Fraud disimpan.")
    else:
        print(f"--- GAGAL ---: Skrip selesai, tapi file {file_fraud} TIDAK ditemukan atau ukurannya 0.")
        print("Ini biasanya terjadi jika 'pydot' masih tidak bisa menemukan 'dot.exe'.")

except Exception as e:
    print(f"\n--- GAGAL MEN-GENERATE DIAGRAM FRAUD (ERROR) ---")
    print(f"ERROR YANG SEBENARNYA ADALAH: {e}")
    print("\nSOLUSI: Pastikan Anda sudah me-RESTART terminal Anda.")

# --- Percobaan Generate Diagram Olist ---
print(f"\nMencoba men-generate diagram Olist ke: {file_olist} ...")
try:
    plot_model(
        model_olist,
        to_file=file_olist,
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        rankdir='LR'
    )
    
    if os.path.exists(file_olist) and os.path.getsize(file_olist) > 0:
        print(f"BERHASIL: Diagram Olist disimpan.")
    else:
        print(f"--- GAGAL ---: Skrip selesai, tapi file {file_olist} TIDAK ditemukan atau ukurannya 0.")

except Exception as e:
    print(f"\n--- GAGAL MEN-GENERATE DIAGRAM OLIST (ERROR) ---")
    print(f"ERROR YANG SEBENARNYA ADALAH: {e}")

print("\nSkrip selesai.")