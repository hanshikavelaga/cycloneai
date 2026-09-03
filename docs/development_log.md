# CycloneAI Development Log (SIH 26070)

This document serves as the master, pin-to-pin log of every development action, file creation, code modification, and algorithm design choice made during the 36-hour build of the CycloneAI platform.

---

## [2026-08-26 09:18] Action 1: Directory Structure Initialization

### 📁 Target Directory
`C:\Users\hansh\Downloads\CycloneAI`

### 🛠️ Action Taken
Created the full, professional enterprise directory structure for the CycloneAI project. This layout separates the system into distinct layers: Machine Learning (`src/models/`), Backend API (`backend/`), Frontend UI (`frontend/`), Database Migration/Seed (`database/`), local object storage mock (`storage/`), and SIH documentation (`docs/`).

### 📂 Directory Paths Created
1. `data/raw/satellite/insat3d/`, `data/raw/satellite/insat3dr/` - Raw satellite channels.
2. `data/raw/cyclone_tracks/imd/`, `data/raw/cyclone_tracks/ibtracs/` - Best-track coordinates.
3. `data/raw/meteorological/` - Environment variables (SST, pressure, wind).
4. `data/processed/satellite/`, `data/processed/tracks/`, `data/processed/meteorological/`, `data/processed/sequences/` - Cleaned/aligned training inputs.
5. `data/metadata/` - Satellite and dataset catalog files.
6. `notebooks/` - Exploratory notebooks.
7. `src/data/`, `src/features/`, `src/models/`, `src/evaluation/`, `src/utils/` - Core Python packages for training.
8. `models/` - Saved model weight checkpoints (`.pth`).
9. `backend/app/api/`, `backend/app/services/`, `backend/app/database/` - FastAPI backend application.
10. `frontend/public/`, `frontend/src/components/`, `frontend/src/pages/`, `frontend/src/services/` - React frontend.
11. `database/migrations/`, `database/seed/` - Database scripts.
12. `storage/satellite/`, `storage/processed/`, `storage/reports/` - Local filesystem object storage.
13. `tests/` - Unit and integration tests.
14. `docs/architecture/`, `docs/SIH/` - Project documentation and pitch materials.

---

## [2026-08-26 09:18] Action 2: Created `.gitignore`

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\.gitignore`

### 🛠️ Action Taken
Created the git ignore configuration file.

### 🧠 Algorithms & Methods
Exclusion pattern matching using glob structures to filter files based on patterns:
- Prevents byte-compiled Python cache files (`__pycache__/`, `*.pyc`) from cluttering commits.
- Excludes the local virtual environment folder (`venv/`) and environment variable storage (`.env`) to protect API keys and database credentials.
- Ignores large local satellite images inside `storage/` and raw dataset folders while retaining directory structures using `!* .gitkeep`.
- Excludes SQLite database files (`*.db`) and node modules (`node_modules/`) to keep the repository lightweight.

---

## [2026-08-26 09:21] Action 3: Created `requirements.txt`

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\requirements.txt`

### 🛠️ Action Taken
Created the pip package requirements manifest file for the Python environment.

### 🧠 Algorithms & Methods
Defines specific, compatible library versions required for the 5-layer AI and backend architecture:
- **FastAPI / Uvicorn**: High-performance asynchronous ASGI web framework.
- **SQLAlchemy**: SQL toolkit and ORM to abstract database queries, supporting both SQLite and PostgreSQL.
- **psycopg2-binary**: PostgreSQL database driver for seamless Neon/Supabase cloud DB connectivity.
- **PyTorch / Torchvision**: Deep learning library for loading convolutional layers and sequence models.
- **NumPy / Pandas**: Multi-dimensional array operations and tabular data synchronization.
- **Scikit-learn**: Evaluation metrics computation (F1, Precision, Recall, MAE, RMSE).
- **Pillow**: Image decoding and preprocessing for satellite bands.
- **Folium**: Leaflet map generation helper.

---

## [2026-08-26 09:21] Action 4: Created `.env.example` and `.env`

### 📄 Target Files
- `C:\Users\hansh\Downloads\CycloneAI\.env.example`
- `C:\Users\hansh\Downloads\CycloneAI\.env`

### 🛠️ Action Taken
Created environment variable templates and active local configurations.

### 🧠 Algorithms & Methods
Implements 12-factor configuration patterns:
- Configures environment-specific variables like `ENV`, `HOST`, and `PORT`.
- Separates database URLs (`DATABASE_URL`) from source code, enabling local SQLite usage for development/fallback and quick configuration of cloud PostgreSQL (Neon/Supabase) by modifying a single line.
- Sets local folder paths for satellite image storage.

---

## [2026-08-26 09:56] Action 5: Created Database Connection Module

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\backend\app\database\connection.py`

### 🛠️ Action Taken
Created the SQLAlchemy database connection module to bootstrap database engines, set session pools, and configure table metadata declarative bases.

### 🧠 Algorithms & Methods
- **Thread Control Isolation**: Conditionally sets `connect_args={"check_same_thread": False}` when using SQLite (`sqlite:///`) to prevent Python database operations from failing due to multi-threaded execution within FastAPI's asynchronous event loops.
- **Session Pooling**: Utilizes SQLAlchemy `sessionmaker` to spawn local session instances on-demand.
- **Dependency Injection**: Implements a `get_db()` Python generator to handle lifecycle session management (`yield db`), ensuring database connections are systematically closed and cleaned up (`db.close()`) after processing web requests.

---

## [2026-08-26 09:57] Action 6: Created Database Models

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\backend\app\database\models.py`

### 🛠️ Action Taken
Created the relational schema mappings to manage cyclone data, tracking metrics, prediction runs, alert states, and cyclogenesis events.

### 🧠 Algorithms & Methods
Mapped five core objects using standard database mapping structures:
1. **`Cyclone`**: Master log representing historical and active cyclone instances (id format: `YYYY_BASIN_NUM`).
2. **`Observation`**: Ground-truth records including coordinates, wind speed (knots), pressure (hPa), Dvorak pattern category, and image references.
3. **`Prediction`**: Predicted track coordinates, intensity estimation values, and confidence scores mapped over standard time offsets (+6h, +12h, +24h, +48h).
4. **`GenesisPrediction`**: Logs tracking cyclogenesis probabilities (0-100%) for monitored oceanic basins.
5. **`Alert`**: Warnings (Genesis, Track, Landfall, Intensification) tagged with colour-coded severity states (Green, Yellow, Orange, Red).

---

## [2026-08-26 09:58] Action 7: Created Pydantic Validation Schemas

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\backend\app\database\schemas.py`

### 🛠️ Action Taken
Created input validation schemas and output serializer models to control API input/output bounds.

### 🧠 Algorithms & Methods
Implements strict runtime checks using Pydantic:
- **`model_config = ConfigDict(from_attributes=True)`**: Configures Pydantic to read raw SQLAlchemy object models and automatically serialize them to JSON.
- **Data Validation & Sanitization**: Validates parameters (floats for latitude/longitude, wind speed ranges, datetime parsing, alert severity checks) on incoming HTTP posts.
- **Output Masking**: Splits schemas into Base, Create, Response, and DetailResponse versions to prevent raw internal structural columns (like password hashes or system indexes) from being leaked through the API.

---

## [2026-08-26 10:00] Action 8: Created & Ran Synthetic Data Generator

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\src\data\dataset_builder.py`

### 🛠️ Action Taken
Created the dataset builder script to synthesize multi-spectral satellite imagery and generate corresponding meteorological tracks, then executed it to seed the SQL database.

### 🧠 Algorithms & Methods
1. **Multi-Spectral Pattern Simulation (Math Formulas)**:
   - Evaluates a polar coordinate grid mapping radius $R = \sqrt{X^2 + Y^2}$ and angle $\theta = \arctan2(Y, X)$.
   - Simulates rotating cloud bands using a sinusoidal spiral formula:
     $$\text{spirals} = \cos(\text{tightness} \cdot R - \text{arms} \cdot \theta)$$
   - Represents the Central Dense Overcast (CDO) as a central Gaussian function:
     $$\text{cdo} = \exp\left(-\frac{R^2}{2\sigma^2}\right)$$
   - Generates distinct structures by combining spirals and CDO:
     * **Eye Pattern**: Clears the center of the CDO using a reverse Gaussian eye mask:
       $$\text{clouds} = (0.7 \cdot \text{cdo} + 0.3 \cdot \text{spirals}) \times \left(1 - \exp\left(-\frac{R^2}{2\sigma_{\text{eye}}^2}\right)\right)$$
     * **Shear Pattern**: Translates the center coordinates of the cloud mass away from the geometric center.
2. **Spectral Band Mapping**:
   - Maps simulated cloud density into multi-spectral bands:
     * **Red Channel (Infrared)**: Cloud top temperatures, with convective zones appearing bright. Cleared center in Eye pattern is mapped to warmer values (darker pixel readings).
     * **Green Channel (Water Vapor)**: Ambient mid-to-high level moisture mapped to a diffused Gaussian background.
     * **Blue Channel (Visible)**: High-frequency spatial detail for daytime cloud morphology.
3. **Database Seeding Execution**:
   - Executes standard SQLAlchemy `create_all()` commands to instantiate relational tables.
   - Populates database records representing:
     * **Cyclone Fani (Historical)**: Chronological observations of coordinate tracks and shifting patterns from Shear to Curved Band, CDO, and Eye.
     * **Cyclone Aila-II (Active)**: Live observations up to the current timestamp and corresponding future track predictions (+6h, +12h, +24h, +48h).
     * **Oceanic Basins Genesis Predictions**: Live formation warnings (Bay of Bengal 78% probability).
     * **System Warnings**: Automated Alert warnings.

---

## [2026-08-26 10:26] Action 9: Fixed ResidualBlock Shortcut Syntax Error

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\src\models\classification\model.py`

### 🛠️ Action Taken
Corrected a compilation error in the `ResidualBlock` class shortcut initialization.

### 🧠 Algorithms & Methods
Removed redundant and corrupted sequential block initialization codes (`nn.Drawing = ...`) in the residual block's projection layer. Restored the standard residual block downsampling mapping:
- If a projection step is required (due to stride changes or dimension mismatches between inputs and outputs), a $1\times1$ convolution (`nn.Conv2d` with kernel size 1) is used to linearly map the spatial dimensions and channel counts.
- Normalizes mapped dimensions with a batch normalization layer (`nn.BatchNorm2d`) before adding them back to the block output.

---

## [2026-08-26 10:29] Action 10: Ran Seeder & Weight Training Initialization

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\src\models\train_all.py`

### 🛠️ Action Taken
Executed the unified model seeder script to initialize models, run a fast training test, and write torch weights (.pth) binaries.

### 🧠 Algorithms & Methods
1. **CNN Weight Generation (multi_task_cnn.pth)**:
   - Feeds a batch of size 16 containing synthetic $3 \times 128 \times 128$ image tensors.
   - Computes a multi-task loss combining Cross-Entropy Loss for Dvorak cloud patterns and Mean Squared Error (MSE) Loss for wind speed/pressure regressions.
   - Runs 2 epochs of Adam optimization. Saves output weights to `models/classification/multi_task_cnn.pth`.
2. **Genesis Temporal Weight Generation (genesis_lstm.pth)**:
   - Feeds time-series observations of 7 environmental variables over a sequence length of 5 steps.
   - Computes Binary Cross Entropy (BCE) Loss against cyclogenesis probability labels.
   - Optimizes weights and saves parameters to `models/genesis/genesis_lstm.pth`.
3. **Track Forecasting Weight Generation (track_lstm.pth)**:
   - Feeds sequence inputs of 4 variables (`[lat, lon, wind_speed, pressure]`) over a history length of 4 steps.
   - Computes MSE regression loss against future targets (+6h, +12h, +24h, +48h).
   - Optimizes parameters and saves them to `models/track_prediction/track_lstm.pth`.

---

## [2026-08-26 10:31] Action 11: Created Spectral Fusion Module

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\src\features\fusion.py`

### 🛠️ Action Taken
Created the spectral fusion helper to blend visual, infrared, and water-vapor satellite channels.

### 🧠 Algorithms & Methods
Implements multi-channel pixel-level fusion using array transformations:
- Decomposes the 3-channel composite image into individual bands: Infrared (IR), Water Vapor (WV), and Visible (VIS).
- **Convective Enhancement (Weighted Alpha Blending)**:
  $$\text{fused}_R = \alpha \cdot \text{IR} + (1 - \alpha) \cdot \text{VIS}$$
  $$\text{fused}_G = 0.5 \cdot \text{WV} + 0.5 \cdot \text{VIS}$$
- **Structural Sharpening**: Applies a conditional thresholding filter on the Blue channel to enhance the cold eyewall boundary contrast:
  $$\text{fused}_B = \begin{cases} 255.0 - \text{IR} & \text{if } \text{IR} > 200 \\ \alpha \cdot \text{VIS} & \text{otherwise} \end{cases}$$

---

## [2026-08-26 10:31] Action 12: Created FastAPI Backend Application

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\backend\app\main.py`

### 🛠️ Action Taken
Coded the FastAPI application routing endpoints, configuring server CORS middlewares, image streaming responses, and seeder databases.

### 🧠 Algorithms & Methods
Implements REST routing endpoints to power the client:
- **CORS Middleware**: Allows cross-origin requests from the React development server.
- **Relational Metadata Queries**: Fetches active and historical storms directly from SQLite/PostgreSQL using SQLAlchemy queries.
- **Dynamic Image Blending (`/api/satellite/fuse/{filename}`)**: Calls the spectral fusion library, performs alpha-blending on-the-fly, buffers bytes in memory using Python's `BytesIO`, and streams the image via `StreamingResponse` to prevent filesystem overhead.
- **Model Inference Routing (`/api/predict/image`)**: Accepts multipart file uploads, processes the temporary file, runs PyTorch CNN inferences, and returns structural patterns and intensities.
- **Trajectory Calculation (`/api/forecast/{cyclone_id}`)**: Runs the track predictor LSTM, saves forecasts to the DB, and returns future coordinate profiles.
- **Historical Replay Engine (`/api/replay/{cyclone_id}/step/{step_idx}`)**: A specialized time-slice endpoint. Returns observations, calculates forecasts, and appends Explainable AI justifications (e.g., central pressure falls, eyewall structures) based ONLY on observations up to `step_idx`.

---

## [2026-08-26 11:28] Action 13: Created React Frontend Dashboard & Interface

### 📄 Target Files
- `C:\Users\hansh\Downloads\CycloneAI\frontend\package.json`
- `C:\Users\hansh\Downloads\CycloneAI\frontend\vite.config.js`
- `C:\Users\hansh\Downloads\CycloneAI\frontend\tailwind.config.js`
- `C:\Users\hansh\Downloads\CycloneAI\frontend\postcss.config.js`
- `C:\Users\hansh\Downloads\CycloneAI\frontend\index.html`
- `C:\Users\hansh\Downloads\CycloneAI\frontend\src\main.jsx`
- `C:\Users\hansh\Downloads\CycloneAI\frontend\src\index.css`
- `C:\Users\hansh\Downloads\CycloneAI\frontend\src\App.jsx`

### 🛠️ Action Taken
Set up the React application structure using Vite, Tailwind CSS, Leaflet Maps, and Recharts, and built the main interactive meteorological command dashboard.

### 🧠 Algorithms & Methods
1. **Interactive Leaflet Mapping (Direct ref Hook integration)**:
   - Avoids package versioning issues by linking standard Leaflet `L` map bindings directly on a React `useRef` container.
   - Clears vector layers (`layersGroupRef.current.clearLayers()`) on state updates to prevent memory leaks and map lag.
   - Draws path trajectories using vector polyline interpolations:
     * **Historical Track**: White polyline connecting observed coordinates.
     * **Forecast Track**: Dashed blue polyline starting at the latest observation coordinates and extending through predicted coordinate points.
   - Places circle markers color-coded to the Dvorak classification (Red=Eye, Orange=CDO, Yellow=Curved Band, Green=Shear).
   - Projects forecast uncertainty envelopes using circular buffer bounds (radius scales inversely with forecast step confidence).
2. **Recharts Dual-Axis Graphing**:
   - Integrates a responsive double-axis chart plot tracking wind speed (left Y-axis) and central atmospheric pressure (right Y-axis) across both historical track logs and predicted forecast windows.
3. **Historical Replay Simulation Loop**:
    - Establishes a step-slider timeline state. Playing the simulation triggers a React `useEffect` interval loop running every 3 seconds, incrementing the step counter, retrieving the partial history, plotting track slices on the map, running the forecasts on the sliced data, and logging Explainable AI justifications dynamically.

---

## [2026-08-26 11:38] Action 14: Created Smart India Hackathon Abstract Documentation

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\docs\SIH\abstract.md`

### 🛠️ Action Taken
Wrote the project abstract, system flow overview, database layouts, and key innovation highlights for the hackathon pitching deck.

### 🧠 Algorithms & Methods
Distills the solution to facilitate evaluation:
- Maps the system to the 5 intelligence layers (Genesis, Identification, Classification, Forecasting, Explainability).
- Diagrams the database and processing flow (INSAT-3D bands $\rightarrow$ data fusion $\rightarrow$ multi-task models $\rightarrow$ API endpoints $\rightarrow$ Leaflet maps).
- Highlights the core innovations: dynamic spectral image fusion, on-the-fly historical simulation playback, and explainable AI logic.

---

## [2026-08-26 11:58] Action 15: Configured Supabase Cloud Database & Connection Pooler

### 📄 Target File
`C:\Users\hansh\Downloads\CycloneAI\.env`

### 🛠️ Action Taken
Configured the active project database connection string to point to a live cloud PostgreSQL database instance hosted in the Asia-Pacific (Mumbai) region on Supabase, and successfully seeded the schemas.

### 🧠 Algorithms & Methods
1. **Network Constraint Isolation (IPv4 Connection Pooling)**:
   - Initial connections using the direct hostname (`db.xdmxzsxnrecnmdmwfwjq.supabase.co` on port `5432`) failed due to university campus network blocks restricting IPv6 routing.
   - Bypassed this routing block by reconfiguring the database connection URL to point to Supabase's transaction connection pooler (`aws-0-ap-south-1.pooler.supabase.com` on port `6543`), which resolves to IPv4.
   - Configured the username parameter to the pooled format: `postgres.xdmxzsxnrecnmdmwfwjq` (which tells the pooler proxy how to route the request to the specific database instance).
2. **Database Cloud Seeding Execution**:
   - Re-executed `dataset_builder.py` over the pooler proxy.
   - Automatically executed DDL migrations (SQL tables instantiation) and inserted the full mock observational paths, prediction horizons, alert messages, and genesis watches directly into the Supabase database.

