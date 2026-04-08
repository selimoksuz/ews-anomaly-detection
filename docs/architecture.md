# EWS Anomaly Detection — Architecture

## Pipeline Flow

```
python cli.py setup    → Oracle tablolarini olustur
python cli.py load     → Sentetik veri uret ve yukle
python cli.py train    → Modeli egit (Oracle → model pickle)
python cli.py score    → Skorla (Oracle → model → Oracle)
python cli.py run      → Train + Score tek komut
python cli.py test     → Oracle'siz lokal test
```

## Folder Structure

```
ews-anomaly-detection/
├── cli.py                    CLI entry point
├── Dockerfile                Container-ready
├── requirements.txt
├── generate_data.py          Sentetik veri uretici
│
├── engine/                   Core pipeline modulleri
│   ├── __init__.py
│   ├── config_loader.py      YAML config/secrets okuyucu
│   ├── oracle_io.py          Oracle read/write/DDL
│   ├── models.py             AE + IF + Mahalanobis
│   ├── scorer.py             Ensemble scoring + explanation
│   └── pipeline.py           Orchestrator (train/score/setup)
│
├── config/
│   ├── pipeline_config.yaml  Tum parametreler (git'e girer)
│   └── secrets.yaml          Oracle credentials (git'e GIRMEZ)
│
├── docs/
│   ├── architecture.md       Bu dosya
│   └── FEATURE_DICTIONARY.md Degisken sozlugu
│
├── logs/                     Calisma loglari (git'e girmez)
├── models/                   Egitilmis model pickle (git'e girmez)
└── tests/                    Test dosyalari
```

## Model Architecture

3-model ensemble:

| Model | Prensip | Agirlik | Ne Yakalar |
|-------|---------|---------|------------|
| Autoencoder | Reconstruction | 0.50 | Degiskenler arasi iliski kirilmasi |
| Isolation Forest | Partition | 0.30 | Global izolasyon, outlier noktalar |
| Mahalanobis (LedoitWolf) | Distance | 0.20 | Kovaryans-aware cok degiskenli uzaklik |

## Feature Layers

| Katman | Adet | Degisim Frekansi |
|--------|------|------------------|
| Anlik | 8 | Her hafta kesin degisir |
| Rolling 4W | 11 | Pencere kayar, haftalik |
| Trend | 9 | Slope/ivme haftalik guncellenir |
| **Toplam** | **28** | |

## Scoring Output

Her musteri icin:
- `anomaly_score`: 0-100 ensemble skor
- `alert_band`: NORMAL / SARI / TURUNCU / KIRMIZI
- `uni_flag_count`: Kac feature bireysel olarak anormal
- `neden`: Top-3 neden (human-readable)
- `detay`: Feature bazinda beklenen vs gerceklesen

## Oracle Tables

| Tablo | Yon | Icerik |
|-------|-----|--------|
| EWS_TRAINING_DATA | Input | Egitim verisi (TRAIN/TEST split) |
| EWS_SCORING_DATA | Input | Bugunun verisi |
| EWS_ALERT_RESULTS | Output | Skor + bant + nedenler |
| EWS_ALERT_DETAILS | Output | Feature bazli detay |
