# 🌀 CycloneAI
### Tropical Cyclone Intelligence & Automated Forecasting Platform
*Smart India Hackathon 2026 — Team Submission*

---

## 📖 Overview

**CycloneAI** is an end-to-end, automated near-real-time tropical cyclone monitoring and forecasting platform designed for the **North Indian Ocean basin** (Bay of Bengal & Arabian Sea).

The system continuously polls authoritative meteorological data sources (GDACS & NOAA), filters active storms for the Indian subcontinent region, and presents live intelligence to meteorologists and disaster response teams through an interactive dashboard.

> **Phase 1 (Current):** Near-real-time ingestion, storm tracking, and operational dashboard.
> **Phase 2 (Planned):** AI-powered trajectory forecasting and intensity classification using trained CNNs and LSTMs.

---

## ✨ Features

| Feature | Status |
|---|---|
| 🔴 Near-real-time GDACS + NOAA live feed polling (every 5 min) | ✅ Live |
| 🗺️ Interactive Leaflet map with cyclone track visualization | ✅ Live |
| 🕐 Live IST clock with UTC→IST frontend conversion | ✅ Live |
| 💾 Supabase PostgreSQL cloud database with duplicate prevention | ✅ Live |
| 🔁 Manual "Sync Now" on-demand trigger | ✅ Live |
| 📊 Recharts wind speed & pressure intensity curves | ✅ Live |
| 🛰️ Satellite image upload + spectral fusion (IR/WV/VIS) | ✅ Live |
| 🤖 Automated Cyclone Genesis probability | ✅ Live |
| 🎬 Historical cyclone replay simulator | ✅ Live |
| 🧠 AI Trajectory Forecast (LSTM) | ⏳ Training |
| 🔬 AI Intensity Classification (Multi-task CNN) | ⏳ Training |
| 📡 INSAT-3D / MOSDAC live satellite integration | 🔲 Planned |

---

## 🏗️ Architecture

```
                       LIVE DATA SOURCES
                  ┌──────────────────────────┐
                  │   GDACS (UN/EC Feed)      │
                  │   NOAA IBTrACS Active     │
                  └────────────┬─────────────┘
                               │
                         Normalize
                               │
                        Region Filter
                    (0–35°N, 40–110°E)
                    ┌───────────┴───────────┐
                    ▼                       ▼
              NORTH INDIAN OCEAN          OTHER
              (Store + Display)          (Log only)
                    │
                    ▼
              Supabase Cloud DB (PostgreSQL, UTC)
                    │
                    ▼
              FastAPI Backend
                    │
                    ▼
              React Dashboard
         (Timestamps displayed in IST)
```

---

## 🗂️ Project Structure

```
CycloneAI/
│
├── backend/
│   └── app/
│       ├── database/
│       │   ├── connection.py
│       │   ├── models.py
│       │   └── schemas.py
│       ├── services/
│       │   ├── gdacs_client.py
│       │   ├── ibtracs_live_client.py
│       │   ├── region_filter.py
│       │   ├── status_cache.py
│       │   ├── live_sync.py
│       │   ├── detection_service.py
│       │   ├── forecast_service.py
│       │   └── genesis_service.py
│       ├── scheduler/
│       │   └── live_scheduler.py
│       └── main.py
│
├── frontend/
│   └── src/
│       └── App.jsx
│
├── src/
│   ├── data/
│   │   ├── dataset_builder.py
│   │   └── spatial_temporal_join.py
│   ├── features/
│   │   └── fusion.py
│   └── models/
│       ├── classification/
│       ├── genesis/
│       ├── track_prediction/
│       └── train_all.py
│
├── backend/scripts/
│   └── test_live_sync.py
│
├── storage/satellite/
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

### Backend
| Package | Purpose |
|---|---|
| FastAPI | REST API server |
| Uvicorn | ASGI production server |
| SQLAlchemy | ORM for PostgreSQL |
| psycopg2-binary | PostgreSQL driver |
| PyTorch | CNN and LSTM model inference |
| Pandas | Tabular data parsing |
| Pillow | Satellite image processing |
| Requests | GDACS/NOAA HTTP polling |

### Frontend
| Package | Purpose |
|---|---|
| React 18 | UI component framework |
| Tailwind CSS | Styling |
| Leaflet.js | Interactive cyclone track map |
| Recharts | Wind/pressure intensity charts |

### Infrastructure
| Service | Purpose |
|---|---|
| Supabase | Managed PostgreSQL cloud database |
| OpenStreetMap | Basemap tile provider (no API key) |
| GDACS | UN/EC live tropical cyclone feed |
| NOAA IBTrACS | Active 7-day storm CSV database |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- A Supabase project

### 1. Clone the Repository
```bash
git clone https://github.com/your-team/cycloneai.git
cd CycloneAI
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```
Edit `.env` with your Supabase credentials:
```env
DATABASE_URL=postgresql://postgres.xxxx:YOUR_PASSWORD@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
SECRET_KEY=your-secret-key
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize the Database
```bash
python src/data/dataset_builder.py
```

### 5. Start the Backend Server
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

### 6. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

### 7. Open the Dashboard
Visit `http://localhost:5173` in your browser.

---

## 🧪 Manual Sync Test

```bash
python backend/scripts/test_live_sync.py
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | /api/health | Server health check |
| GET | /api/system/status | Live feed sync status & logs |
| POST | /api/system/sync | Trigger manual sync |
| GET | /api/cyclones/active | List active cyclones |
| GET | /api/cyclones/historical | List historical cyclones |
| GET | /api/cyclones/{id} | Cyclone detail + observations |
| POST | /api/forecast/{id} | Run LSTM track forecast |
| GET | /api/genesis | Genesis probability predictions |
| GET | /api/alerts | Active system alerts |
| POST | /api/predict/image | Classify satellite image |

---

## 🌊 Live Data Sources

| Source | Update Cadence |
|---|---|
| GDACS Tropical Cyclone RSS Feed | ~6 minutes |
| NOAA IBTrACS Active CSV | 3x per week |

> **Region Filter**: Only storms within Lat 0°N–35°N, Lon 40°E–110°E are stored.
> **Uniqueness Constraint**: source + storm_id + timestamp prevents duplicate observations.

---

## 👥 Team

| Member | Role |
|---|---|
| Hanshika | Integration Lead — Backend Architecture, Database, Live Ingestion |
| Meghana | Team 1 — IBTrACS Track Data & DB Schema |
| Vasavi | Team 2 — ERA5 Weather Variables |
| Chandini | Team 2 — ERA5 Data Processing |
| Hasini | Team 3 — Satellite Data & Genesis AI Design |
| Goutami | Team 3 — HURSAT/INSAT Image Processing |

---

## 🗺️ Roadmap

- [x] Near-real-time GDACS/NOAA ingestion pipeline
- [x] Supabase cloud database with duplicate prevention
- [x] React dashboard with live IST clock and sync indicators
- [x] OpenStreetMap basemap integration
- [x] Satellite spectral fusion (IR/WV/VIS)
- [x] Historical cyclone replay simulator
- [ ] IBTrACS + ERA5 + Satellite historical dataset integration
- [ ] Feature engineering pipeline
- [ ] Multi-task CNN training
- [ ] Track prediction LSTM training
- [ ] Connect trained AI models to live ingestion pipeline
- [ ] INSAT-3D / MOSDAC direct satellite feed integration

---

## 📜 License

This project was developed for the Smart India Hackathon 2026. All rights reserved by the team.

---

Built with ❤️ by Team CycloneAI
