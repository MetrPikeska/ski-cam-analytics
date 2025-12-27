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

- **Python 3.11** (nebo novější)
- **FFmpeg** - musí být nainstalovaný a dostupný v PATH

### 2. Instalace Python závislostí

```bash
cd backend
pip install -r requirements.txt
```

### 3. Instalace FFmpeg

#### Windows:
1. Stáhněte FFmpeg z https://ffmpeg.org/download.html
2. Rozbalte do složky (např. `C:\ffmpeg`)
3. Přidejte `C:\ffmpeg\bin` do PATH
4. Ověřte: `ffmpeg -version`

#### Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS:
```bash
brew install ffmpeg
```

### 4. YOLO Model

Aplikace potřebuje YOLO ONNX model pro detekci osob.

**Umístění:** `models/yolo.onnx`

**Jak získat model:**

Máte několik možností:

#### A) YOLOv8 (doporučeno)
```bash
pip install ultralytics
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx')"
```
Vyexportovaný model přesuňte do `models/yolo.onnx`

#### B) YOLOv5
1. Stáhněte pre-trained model: https://github.com/ultralytics/yolov5/releases
2. Export do ONNX:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
python export.py --weights yolov5n.pt --include onnx
```

#### C) Vlastní model
- Použijte jakýkoliv YOLO model trénovaný na COCO datasetu
- Zajistěte že class 0 = person
- Export do ONNX formátu

**Poznámka:** Do `models/` složky přidejte `.gitignore`:
```
*.onnx
```

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

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Nebo:
```bash
cd backend
python -m app.main
```

Server běží na: **http://localhost:8000**

### 2. Otevřete dashboard v prohlížeči

```
http://localhost:8000
```

### 3. Spusťte analýzu

Klikněte na tlačítko **"▶️ START ANALÝZY"** v dashboardu.

Pipeline se spustí a začne zpracovávat stream.

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

### FFmpeg stream nefunguje
- Ověřte že FFmpeg je nainstalován: `ffmpeg -version`
- Zkuste stream ručně: `ffmpeg -i https://stream.teal.cz/hls/cam273.m3u8 -t 5 test.mp4`
- Zkontrolujte firewall / síťové připojení

### Model nenalezen
- Ověřte že `models/yolo.onnx` existuje
- Zkontrolujte cestu v `config.py`

### Nízké FPS
- Snižte `FFMPEG_FPS` v config (např. na 4-6)
- Snižte rozlišení (`FRAME_WIDTH`, `FRAME_HEIGHT`)
- Použijte menší YOLO model (yolov8n, yolov5n)

### Špatná detekce
- Zvyšte/snižte `CONF_THRESHOLD` (0.3-0.6)
- Nastavte ROI na relevantní oblast
- Zkuste jiný YOLO model

### Pipeline se automaticky spouští
- **Nemělo by se stávat!** Pipeline se spouští pouze přes `/api/pipeline/start`
- Zkontrolujte že jste neupravili `main.py` lifespan

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
