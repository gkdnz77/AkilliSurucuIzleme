# 🚗 Akıllı Sürücü İzleme Sistemi

Gerçek zamanlı kamera görüntüsü üzerinden sürücünün yorgunluk, dikkat dağınıklığı ve güvenlik durumunu tespit eden yapay zeka destekli sistem.

## 👥 Proje Ekibi

- **Gökdeniz Sağlam**
- **Devran Demir**

## 🎯 Özellikler

- 🎥 Gerçek zamanlı kamera ile yüz tanıma ve göz takibi
- 😴 Yorgunluk ve uykululuk tespiti
- 👁️ Göz yönü analizi ve dikkat dağınıklığı uyarıları
- 📱 Telefon kullanımı algılama
- 🧠 AI tabanlı duygu analizi
- 📊 Oturum kayıtları ve detaylı istatistikler
- 📄 PDF rapor oluşturma
- 🌓 Karanlık/Aydınlık tema desteği

## 📋 Gereksinimler

```bash
Python 3.10+
OpenCV
MediaPipe
NumPy
Flask
ReportLab (PDF için)
```

## 🚀 Kurulum

1. **Projeyi klonlayın:**
```bash
git clone https://github.com/kullaniciadi/akilli-surucu-izleme.git
cd akilli-surucu-izleme
```

2. **Sanal ortam oluşturun:**
```bash
conda create -n gorsel python=3.10
conda activate gorsel
```

3. **Gerekli paketleri yükleyin:**
```bash
pip install opencv-python mediapipe numpy flask reportlab pillow
```

4. **Projeyi başlatın:**
```bash
# Web arayüzü için
jupyter notebook web_app.ipynb

# Veya Python scripti olarak
python driver_system.py
```

## 📁 Proje Yapısı

```
EmotionRecognition/
│
├── driver_system.ipynb          # Ana sürücü izleme sistemi
├── web_app.ipynb               # Web arayüzü ve yönetim paneli
│
├── emotion_model.pkl           # Eğitilmiş duygu modeli
├── normalization_params.npz    # Model normalizasyon parametreleri
├── driver_profile.npy          # Kayıtlı sürücü profili
│
├── sessions/                   # Oturum kayıtları (JSON)
│   └── surucu_oturumu_*.json
│
├── DejaVuSans.ttf             # PDF için font dosyası (opsiyonel)
│
└── README.md
```

## 💻 Kullanım

### Web Arayüzü

1. `web_app.ipynb` dosyasını açın
2. Tüm hücreleri çalıştırın
3. Tarayıcıda `http://localhost:5000` adresine gidin
4. Sürücü bilgilerinizi girin
5. "Sürüşü Başlat" butonuna tıklayın

### Sürücü Sistemi

- **q**: Programı kapat ve oturumu kaydet
- **p**: Ana sürücü profilini kaydet/güncelle
- **r**: İstatistikleri sıfırla
- **t**: Test modunu başlat

## 📊 Özellik Detayları

### Gerçek Zamanlı Tespit
- Göz açıklık oranı (EAR)
- Baş pozisyonu (Yaw, Pitch, Roll)
- Göz yönü (iris tracking)
- Ağız açıklığı (esnerme tespiti)

### Risk Skoru Hesaplama
```
Risk = (Uykululuk × 0.45) + (Dikkat Dağınıklığı × 0.45) + (Yorgunluk × 0.10)
```

### Oturum Verileri
Her oturum için:
- Toplam süre
- Ortalama göz açıklığı
- Uykululuk yüzdesi
- Dikkat dağınıklığı yüzdesi
- Duygu dağılımı
- Telefon kullanım istatistikleri

## 🔧 Yapılandırma (Opsiyonel)

Proje, ek özellikler için ortam değişkenleri kullanır. Bunlar **zorunlu değildir**:

```bash
# Telegram bildirimleri (opsiyonel)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# AI Chatbot (opsiyonel)
ANTHROPIC_API_KEY=your_api_key

# Acil durum ayarları
EMERGENCY_EYES_CLOSED_SECONDS=10
EMERGENCY_COOLDOWN_SECONDS=120
```

> **Not:** Bu özellikler olmadan da proje tam olarak çalışır. Sadece temel sürücü izleme için yukarıdaki değişkenlere ihtiyaç yoktur.

## 📸 Ekran Görüntüleri

### Ana İzleme Ekranı
- Gerçek zamanlı yüz takibi
- Göz açıklık grafiği
- Duygu analizi
- Uyarı sistemleri

### Web Panel
- Oturum geçmişi
- Karşılaştırma araçları
- PDF rapor indirme
- İstatistik kartları

## 🤝 Katkıda Bulunma


