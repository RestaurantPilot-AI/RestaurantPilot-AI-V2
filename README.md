# RestaurantPilot

This repository contains an invoice ingestion and OCR pipeline (Streamlit UI, extraction, processing, and MongoDB storage). The project uses Tesseract / EasyOCR models and expects a running MongoDB instance and a `.env` file with configuration.

**This README focuses on installation and getting the project running locally (Windows).**

**Prerequisites**
- **Python:** 3.10+ recommended. Use a virtual environment.
- **MongoDB:** A running MongoDB server (local or remote). Community edition is fine.
- **Tesseract OCR:** Required by `pytesseract`/Tesseract pathway. The repo includes a `tesseract_models` folder but installing system Tesseract is recommended for Windows.
- **Git:** optional, to clone the repo.

**Included files to check before running**
- `requirements.txt`: Python dependencies. Install them into a virtual environment.
- `.env`: environment variables (see example below).
- `src/extraction/tesseract_models/`: shipped tesseract model files and a tesseract executable placeholder.

**Setup — Windows (PowerShell)**
1. Open PowerShell and create & activate a virtual environment:

```powershell
python -m venv .venv
# PowerShell activation
.\.venv\Scripts\Activate.ps1
``` 

2. Install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Install Tesseract (Windows)
- Download the Windows installer from: https://github.com/UB-Mannheim/tesseract/wiki
- Install (default path often: `C:\Program Files\Tesseract-OCR`).
- Ensure `tesseract.exe` is on your PATH, or set `TESSDATA_PREFIX` / set the `tesseract_cmd` in `src/extraction/config.py` to point to the installed `tesseract.exe`.

Example PowerShell to add Tesseract to PATH for current session (adjust path as needed):

```powershell
$env:Path += ";C:\Program Files\Tesseract-OCR"
$env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR\tessdata"
```

4. MongoDB
- Install and run MongoDB Community edition locally, or use a hosted MongoDB Atlas cluster.
- Default connection string used by the repo (in `src/storage/db_init.py`) is `mongodb://localhost:27017/`.

5. Create or edit `.env` at the project root (example below).

**Recommended `.env` (example)**
Create a file named `.env` in the project root with the following variables (example values):

```text
# MongoDB
MONGODB_URI=mongodb://localhost:27017/
DB_NAME=invoice_processing_db

# Google / API keys used in processing (if applicable)
GOOGLE_API_KEY=
GEMINI_API_KEY=

# Optional: point to system tesseract if not using shipped executable
# TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

Notes:
- The app loads environment variables via `python-dotenv`. Files like `src/storage/db_init.py` and `src/storage/database.py` read `MONGODB_URI` and `DB_NAME`.
- Provide API keys only if you intend to use features that call Google/Gemini APIs.

**Models & Large Files**
- Some OCR models and training data are stored under `src/extraction/dependencies/` and `src/extraction/tesseract_models/`. These are included in the repo; confirm they exist after cloning.

**Run the application (development)**
- The project uses Streamlit for the UI. To start the app locally (from project root):

```powershell
streamlit run app.py
```

- Alternative: some environments include `main.py` — but the primary UI entry observed is `app.py`.

**Initialize the database (optional)**
- There is a `src/storage/db_init.py` helper that reads `MONGODB_URI` and `DB_NAME` and attempts to connect. Run it if you need to verify connectivity or create initial collections:

```powershell
python src\storage\db_init.py
```

**Troubleshooting**
- If OCR calls fail with `tesseract` not found, either install Tesseract system-wide or update `PATH` / `src/extraction/config.py` `tesseract_cmd` to the correct executable path.
- If MongoDB connection fails, verify `MONGODB_URI` and that the MongoDB service is running. Try connecting with `mongo` shell or Compass.
- If you see missing model/data errors, confirm `src/extraction/dependencies/` and `src/extraction/tesseract_models/` contain the expected files.

**Useful Commands (PowerShell)**
- Create & activate venv:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

- Install deps:

```powershell
pip install -r requirements.txt
```

- Run app:

```powershell
streamlit run app.py
```

**Next steps & Notes**
- After verifying the environment and `.env` values, upload or place sample invoices in the configured ingestion path and use the Streamlit UI to run extraction.
- If you want, I can add a small `README_ENV_TEMPLATE` or `.env.example` file into the repo. Want me to add that now?

---
License and notices are present in `LICENSE.txt` and `THIRD_PARTY_NOTICES.txt` in the repository root.

