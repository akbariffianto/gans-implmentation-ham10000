## Prasyarat
Pastikan sudah menginstal **Python 3.8+** di sistem Anda.

### 1. Buat Virtual Environment
Langkah ini penting agar *library* proyek skripsi Anda tidak bentrok dengan proyek Python lainnya. Jalankan perintah ini di terminal:

```
python -m venv venv
```
### 2. Aktivasi Virtual Environment

Aktifkan environment yang baru saja dibuat. Tanda jika berhasil adalah muncul tulisan (venv) di sebelah kiri command prompt.

Windows:
```
.\venv\Scripts\activate
```
macOS/Linux:
```
source venv/bin/activate
```
### 3. Instal Dependencies
Pastikan file requirements.txt sudah tersedia di folder proyek, lalu jalankan:
```
pip install -r requirements.txt
```
### 4.Menjalankan Aplikasi
Setelah semua pustaka terinstal, jalankan aplikasi Streamlit dengan perintah berikut:

```
streamlit run app.py
```