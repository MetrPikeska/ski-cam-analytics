# Ski Cam Analytics 🎿

MVP aplikace pro počítání lidí v lyžařském areálu z HLS video streamu.

**Klíčová vlastnost:** Aplikace běží pouze po ručním spuštění (START/STOP), není to 24/7 daemon.

---

## 📋 Co aplikace dělá

- **Načítá HLS stream** přes FFmpeg
- **Detekuje osoby** pomocí YOLO ONNX modelu (CPU inference)
- **Trackuje osoby** jednoduchým SORT-like trackerem
- **Počítá metriky:**
  - **Occupancy** - aktuální počet lidí ve scéně
  - **Line Crossing** - kolik osob překročilo definovanou čáru (vlek/brána)
- **Ukládá agregace** do SQLite databáze (po minutách)
- **Zobrazuje dashboard** v prohlížeči s real-time aktualizacemi

---

## 🚀 Instalace

### 1. Požadavky

- **Python 3.10+** (testováno na Python 3.10)
- **FFmpeg** - **KRITICKÁ ZÁVISLOST** pro načítání HLS streamu

### 2. Instalace FFmpeg

**⚠️ DŮLEŽITÉ: FFmpeg musí být nainstalován PŘED spuštěním aplikace!**

#### Windows (doporučeno - winget):
```powershell
winget install --id Gyan.FFmpeg -e --accept-source-agreements
```

Po instalaci **restartujte PowerShell** nebo aktualizujte PATH:
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

Ověření:
```powershell
ffmpeg -version
```

#### Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS:
```bash
brew install ffmpeg
```

### 3. Instalace Python závislostí

```bash
cd backend
pip install -r requirements.txt
```

**Poznámka:** Požadavky zahrnují:
- `numpy>=2.1.3` (Python 3.10+ kompatibilní verze)
- `opencv-python>=4.10.0`
- `onnxruntime>=1.20.1`
- `fastapi>=0.115.5`
- `uvicorn>=0.34.0`

### 4. YOLO Model

Aplikace potřebuje YOLO ONNX model pro detekci osob.

**Umístění:** `models/yolo.onnx`

**Jak získat model (nejjednodušší způsob):**

1. Stáhněte YOLOv8n ONNX model přímo:
   - https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx

2. Uložte jako `models/yolo.onnx` v kořenovém adresáři projektu

**Alternativně - export z PyTorch:**

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx')"
mv yolov8n.onnx models/yolo.onnx
```

**Poznámka:** Model `yolo.onnx` je ignorován gitem (.gitignore)

---

## ⚙️ Konfigurace

Všechny parametry jsou v **`backend/app/config.py`**:

```python
# Stream URL
STREAM_URL = "https://stream.teal.cz/hls/cam273.m3u8"

# Processing parameters
FFMPEG_FPS = 8  # Kolik FPS zpracovávat (nižší = menší zátěž)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection
CONF_THRESHOLD = 0.4  # Confidence threshold (0.3-0.5)
IOU_THRESHOLD = 0.45

# Tracking
TRACKER_MAX_AGE = 30  # Max frames bez detekce
TRACKER_MIN_HITS = 3  # Min počet hitů pro confirmed track

# ROI (Region of Interest) - omezí detekci jen na tuto oblast
ROI_RECT = None  # Příklad: (100, 150, 540, 450)

# Line Crossing - čára pro počítání průchodů
LINE_CROSSING = None  # Příklad: [(200, 300), (440, 300)]
```

### Jak nastavit ROI a Line Crossing?

1. Spusťte aplikaci
2. Prohlédněte si stream / snímky
3. Určete souřadnice (můžete použít screenshot + image editor)
4. Nastavte v `config.py`:
   - **ROI_RECT**: `(x1, y1, x2, y2)` - levý horní a pravý dolní roh
   - **LINE_CROSSING**: `[(x1, y1), (x2, y2)]` - dva body definující čáru
5. Restartujte pipeline (STOP → START)

---

## 🎯 Spuštění

### 1. Spusťte backend server

**Windows PowerShell (doporučeno):**
```powershell
cd backend
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$env:PYTHONPATH = "C:\Users\<VaseJmeno>\Documents\GitHub\ski-cam-analytics\backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Linux/macOS:**
```bash
cd backend
export PYTHONPATH="${PWD}"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server běží na: **http://localhost:8000**

**Poznámky:**
- `--reload` zapíná auto-restart při změnách kódu (vhodné pro vývoj)
- FFmpeg **musí být v PATH** (viz instalační sekce)
- Pipeline se **NESPOUŠTÍ automaticky** při startu serveru

### 2. Otevřete dashboard v prohlížeči

```
http://localhost:8000
```

### 3. Spusťte analýzu

Klikněte na tlačítko **"▶️ START ANALÝZY"** v dashboardu.

Pipeline se spustí a začne zpracovávat stream:
1. Načte YOLO model (~6MB YOLOv8n)
2. Spustí FFmpeg pro čtení HLS streamu
3. Začne detekovat a trackovat osoby
4. Zobrazí live video s žlutými bounding boxy

**První spuštění může trvat 5-10 sekund** (načítání modelu).

### 4. Zastavte analýzu

Klikněte na tlačítko **"⏹️ STOP ANALÝZY"**.

### 5. Vypnutí serveru

Ukončete proces serveru (Ctrl+C v terminálu).

---

## 📊 API Endpoints

### Status
```http
GET /api/status
```
Vrátí stav pipeline (běží/neběží, FPS, uptime).

### Start Pipeline
```http
POST /api/pipeline/start
```
Spustí analýzu.

### Stop Pipeline
```http
POST /api/pipeline/stop
```
Zastaví analýzu.

### Metriky
```http
GET /api/metrics/latest
```
Aktuální metriky (occupancy, crossings).

```http
GET /api/metrics/timeseries?minutes=60
```
Časová řada za posledních N minut.

### WebSocket
```
WS /ws/live
```
Real-time push metrik každou sekundu.

---

## 🗂️ Struktura projektu

```
ski-cam-analytics/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI server
│   │   ├── pipeline.py        # Hlavní processing pipeline
│   │   ├── ffmpeg_source.py   # FFmpeg video reader
│   │   ├── detector_onnx.py   # YOLO ONNX detektor
│   │   ├── tracker.py         # SORT-like tracker
│   │   ├── analytics.py       # Počítání metrik
│   │   ├── storage.py         # SQLite storage
│   │   ├── config.py          # Konfigurace
│   │   └── models.py          # Pydantic modely
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── models/
│   └── yolo.onnx              # YOLO model (přidejte ručně)
├── data/
│   └── metrics.db             # SQLite databáze (vytvoří se auto)
└── README.md
```

---

## 🔧 Troubleshooting

### ❌ Chyba: "FFmpeg není nainstalován nebo není v PATH!"
**Příčina:** FFmpeg není dostupný v systémové PATH proměnné.

**Řešení:**
1. Nainstalujte FFmpeg (viz instalační sekce)
2. **Windows:** Restartujte PowerShell terminál po instalaci
3. Ověřte: `ffmpeg -version`
4. Pokud instalace proběhla v aktuální session, aktualizujte PATH:
   ```powershell
   $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
   ```

### ❌ Chyba: "Failed to start pipeline" (500 error)
**Možné příčiny:**
1. Model `models/yolo.onnx` neexistuje nebo má špatný název
2. FFmpeg není v PATH
3. Stream je nedostupný (síť, firewall)

**Řešení:**
1. Ověřte existenci modelu: `ls models/yolo.onnx`
2. Zkontrolujte FFmpeg: `ffmpeg -version`
3. Otestujte stream ručně:
   ```bash
   ffmpeg -i https://stream.teal.cz/hls/cam273.m3u8 -t 5 test.mp4
   ```

### ❌ "No frame available" na dashboardu
**Příčina:** Pipeline neběží nebo neprodukuje frames.

**Řešení:**
1. Klikněte na START ANALÝZY
2. Zkontrolujte server logy v terminálu
3. Ověřte že stream funguje (viz výše)

### ❌ Server crashuje při POST /api/pipeline/start
**Příčina:** Bug v lifespan manageru (opraveno ve verzi 1.1).

**Řešení:**
- Aktualizujte kód (`git pull`)
- Ověřte že `main.py` obsahuje `global broadcast_task`

### 🐌 Nízké FPS / pomalé zpracování
**Řešení:**
- Snižte `FFMPEG_FPS` v config (např. na 4-6)
- Snižte rozlišení (`FRAME_WIDTH=480`, `FRAME_HEIGHT=360`)
- Použijte menší YOLO model (yolov8n)

### 🎯 Špatná nebo žádná detekce
**Řešení:**
- Zvyšte/snižte `CONF_THRESHOLD` (0.3-0.6)
- Nastavte ROI na relevantní oblast v `config.py`
- Zkuste jiný YOLO model (yolov8s pro vyšší přesnost)

### 🌙 Noční provoz (tmavé video)
**Poznámka:** YOLO model detekuje špatně za tmy.

**Řešení:**
- Použijte model trénovaný na nočních datech
- Nebo vypněte analýzu v noci (není to daemon, spouští se ručně)

---

## 🚀 Co zlepšit pro ostrý provoz

Toto je **MVP** - jednoduchý prototyp pro testování. Pro produkční nasazení zvažte:

### Performance
- [ ] **GPU inference** - přidat CUDA support pro ONNX Runtime
- [ ] **Optimalizovaný tracker** - použít DeepSORT nebo ByteTrack
- [ ] **Async processing** - oddělení čtení frames a inference
- [ ] **Frame buffer** - lepší handling při výpadcích streamu

### Robustnost
- [ ] **Auto-restart** při pádu streamu
- [ ] **Monitoring** - healthchecky, alerting
- [ ] **Logování** - strukturované logy, rotace
- [ ] **Error handling** - graceful degradation

### Features
- [ ] **Noční režim** - detekce a filtrování za tmy
- [ ] **Heatmapa** - vizualizace pohybu lidí
- [ ] **Kalibrace** - automatické nastavení ROI/line
- [ ] **Multi-camera** - podpora více streamů
- [ ] **Export dat** - CSV/JSON export pro analýzu
- [ ] **Alerting** - notifikace při vysoké occupancy

### Deployment
- [ ] **Docker** - containerizace aplikace
- [ ] **Systemd service** - auto-start při boot
- [ ] **Reverse proxy** - Nginx + SSL
- [ ] **Authentication** - zabezpečení dashboardu

### Analytics
- [ ] **PostgreSQL** - místo SQLite pro větší data
- [ ] **Grafana** - pokročilé grafy a dashboardy
- [ ] **ML predikce** - předpovídání occupancy
- [ ] **Statistiky** - denní/týdenní reporty

---

## 📝 Poznámky

- **Pouze lokální provoz** - aplikace běží na localhost
- **Ručně spouštěná** - žádné automatické spouštění
- **SQLite storage** - pouze agregované metriky, ne video
- **CPU inference** - dostatečné pro testování, pro produkci GPU
- **Bez autentizace** - zabezpečte před veřejným přístupem

---

## 📄 Licence

MIT License - použijte podle potřeby.

---

## 🤝 Podpora

Pro otázky a problémy vytvořte issue nebo kontaktujte vývojáře.

---

**Enjoy skiing! ⛷️🏔️**
