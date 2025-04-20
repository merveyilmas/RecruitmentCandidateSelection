# İşe Alımda Aday Seçimi: SVM ile Başvuru Değerlendirme

## Proje Hakkında
Bu proje, yazılım geliştirici pozisyonu için başvuran adayların tecrübe yılı ve teknik sınav puanına göre işe alınıp alınmamasını tahmin eden bir makine öğrenmesi modeli sunar.

## API Dokümantasyonu

### Temel Bilgiler
- **Base URL**: `http://localhost:8000`
- **API Versiyonu**: 1.0.0

### Endpoint'ler

#### 1. Kök Endpoint
```
GET /
```
Ana sayfa endpoint'i. API'nin çalıştığını doğrulamak için kullanılır.

**Response**:
```json
{
    "message": "Welcome to Candidate Selection API"
}
```

#### 2. Aday Tahmini
```
POST /predict
```
Adayın işe alınıp alınmayacağını tahmin eder.

**Request Body**:
```json
{
    "experience_years": 3.5,
    "technical_score": 75.0
}
```

**Response**:
```json
{
    "prediction": 0,
    "probability": 0.85
}
```
- `prediction`: 0 (İşe alındı) veya 1 (İşe alınmadı)
- `probability`: Tahminin güven skoru (-1 ile 1 arası)

#### 3. Model Bilgileri
```
GET /model_info
```
Kullanılan modelin parametrelerini ve özelliklerini gösterir.

**Response**:
```json
{
    "model_type": "SVC",
    "kernel": "linear",
    "C": 1.0,
    "gamma": "scale"
}
```

### Örnek Kullanım

#### Python ile API Kullanımı
```python
import requests

# API endpoint
BASE_URL = "http://localhost:8000"

# Tahmin yapma
def predict_candidate(experience_years: float, technical_score: float):
    data = {
        "experience_years": experience_years,
        "technical_score": technical_score
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    return response.json()

# Örnek kullanım
result = predict_candidate(3.5, 75.0)
print(result)
```

#### cURL ile API Kullanımı
```bash
# Tahmin yapma
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"experience_years": 3.5, "technical_score": 75.0}'

# Model bilgilerini alma
curl "http://localhost:8000/model_info"
```

### Hata Kodları
- `200`: Başarılı
- `400`: Geçersiz istek
- `500`: Sunucu hatası

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. API'yi başlatın:
```bash
uvicorn app.main:app --reload
```

3. Tarayıcınızda Swagger UI'ı açın:
```
http://localhost:8000/docs
```

## Model Detayları
- Algoritma: Support Vector Machine (SVM)
- Kernel: Linear
- Özellikler: Tecrübe Yılı, Teknik Sınav Puanı
- Etiketler: 0 (İşe alındı), 1 (İşe alınmadı)

## Geliştirici Notları
- Model eğitimi için en az 200 örnek veri kullanılmaktadır
- Veriler otomatik olarak oluşturulur ve `data/processed/candidate_data.csv` dosyasında saklanır
- Model parametreleri `models/saved` klasöründe saklanır 