# Smart India Hackathon 2026 Submission

* **Problem Statement ID:** 26070
* **Problem Statement Title:** To develop an Artificial Intelligence (AI) / Machine Learning (ML) based system for identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data.
* **Project Title:** CycloneAI — AI-Powered Tropical Cyclone Detection, Classification & Forecasting Platform
* **Theme:** Smart Education / Disaster Management
* **Department:** India Meteorological Department (IMD), Ministry of Earth Sciences (MoES)
* **Category:** Software

---

## 1. Executive Summary

Tropical cyclones are high-impact weather hazards that require rapid and precise monitoring. Traditionally, estimating cyclone structures and track trajectories has relied heavily on the manual Dvorak technique and computationally heavy numerical weather prediction (NWP) models. 

**CycloneAI** addresses this challenge by establishing an end-to-end automated artificial intelligence platform. Using multi-spectral satellite imagery (combining Infrared, Water Vapor, and Visible bands from INSAT-3D/3DR meteorological satellites) integrated with geographical best-track logs, the platform executes five intelligence tasks:
1. **Genesis Prediction**: Evaluates time-series observations of sea surface temperature (SST), atmospheric humidity, and convection structures to predict the probability of cyclogenesis before a low-pressure system organizes.
2. **Structural Classification**: Classifies storm organization into standard Dvorak patterns (Curved Band, Shear, CDO, Eye) using a multi-task Convolutional Neural Network (CNN).
3. **Intensity Estimation**: Estimates continuous parameters (maximum sustained wind speeds in knots and central pressure in hPa).
4. **Trajectory Forecasting**: Predicts future tracks (+6h, +12h, +24h, +48h) by blending LSTM sequence models with physics-based persistence climatology vectors.
5. **Explainable Decision Support**: Automatically extracts physical metrics and image features to present natural-language explanations of model forecasts.

---

## 2. System Architecture

The platform separates large data blocks from metadata query layers to ensure high query speeds:
* **Large Image Storage**: Stacked multi-spectral satellite matrices are stored in local object/file storage (`storage/satellite/`).
* **Relational Database**: PostgreSQL / SQLite stores track logs, predicted paths, genesis basin states, and system warnings.
* **Backend Engine (FastAPI)**: Serves prediction results, blends satellite channels on-the-fly, and updates forecast databases.
* **Frontend Interface (React + Tailwind + Leaflet)**: Renders interactive maps showing cyclone histories, forecast uncertainty cones, and Dvorak intensity graphs.

```
┌──────────────────┐      ┌──────────────────┐
│ INSAT-3D (ISRO)  │      │   IMD Tracks /   │
│ Satellite Bands  │      │   Meteorology    │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         ▼                         ▼
┌────────────────────────────────────────────┐
│              DATA FUSION ENGINE            │
└────────────────────┬───────────────────────┘
                     ▼
┌────────────────────────────────────────────┐
│            MULTI-TASK AI PIPELINE          │
│  - Genesis LSTM    - Classification CNN   │
│  - Intensity CNN   - Track Prediction LSTM │
└────────────────────┬───────────────────────┘
                     ▼
┌────────────────────────────────────────────┐
│            FASTAPI BACKEND APIS            │
│  - Inference APIs   - Fusion Composites     │
│  - Replay Engine    - Database Connectors  │
└────────────────────┬───────────────────────┘
                     ▼
┌────────────────────────────────────────────┐
│            REACT FRONTEND PORTAL           │
│  - Interactive Map  - Recharts Dashboard   │
│  - Replay Slider    - Diagnostic Upload    │
└────────────────────────────────────────────┘
```

---

## 3. Key Innovations & Hackathon Showstoppers

1. **Spectral Channel Fusion**: Built-in blending sliders allow users to fuse Infrared (IR) and Water Vapor (WV) channels dynamically. This highlights high-altitude convective storms overlaying moisture currents, revealing structure features (like the eyewall) that are obscured in single-band views.
2. **Historical Replay Simulator (Interactive Pitch)**: Rather than presenting a static slide show, the platform features a "Historical Replay" player. Judges can select historical storms (like Cyclone Fani) and step through the timeline. The AI runs forecasts at each step using *only* information available up to that hour, comparing predictions side-by-side with the actual track.
3. **Explainable AI (XAI)**: The platform explains *why* the AI makes predictions. It presents textual rationales (e.g. *"Intensity increasing due to falling central pressure (-8 hPa/6h) and organization of central dense overcast clouds"*).
4. **Resilient Local-First Database Design**: The backend uses SQLAlchemy, allowing the platform to run offline using a local SQLite database file, but scales immediately to a production PostgreSQL database (Supabase/Neon) by changing a single environment variable.
