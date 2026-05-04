# Universal Text-to-SQL System

This project provides a FastAPI app that converts natural language questions into SQL and executes them against uploaded or connected databases.

## What Happens With Adapters

Model adapters are stored in `training/artifacts`.

To keep the repository size manageable, adapters can be distributed as a Google Drive zip. The app now auto-handles this:

1. On startup, the app checks whether expected adapter folders exist.
2. If missing, it downloads `training/artifacts.zip` from Google Drive.
3. It extracts the zip at the project root so `training/artifacts/...` is restored.

Configured source:

- Google Drive file id: `1vfhisRUh-LCl7HBj6uPAQ6bExlCqN6HU`

## Prerequisites

- Python 3.10
- Windows PowerShell (for the provided PowerShell commands)
- Internet access (first run if adapters are not present locally)

## Setup

1. Start the app with the launcher script (recommended):

```powershell
.\run_app.ps1
```

This script automatically:

- Creates `.venv` (if missing)
- Upgrades pip
- Installs dependencies from `requirements.txt`
- Starts Uvicorn on port `8000`

2. Optional script flags:

```powershell
.\run_app.ps1 -Port 9000
.\run_app.ps1 -NoReload
.\run_app.ps1 -SkipInstall
.\run_app.ps1 -VenvPath ".venv310-gpu"
```

3. Manual setup (fallback):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run The App

Recommended:

```powershell
.\run_app.ps1
```

Manual:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- UI: `http://127.0.0.1:8000/`
- Health: `http://127.0.0.1:8000/health`
- Model status: `http://127.0.0.1:8000/models/status`

## Typical Usage

1. Upload a `.db`, `.sqlite`, `.sqlite3`, or `.duckdb` file using the UI.
2. Ask a natural-language question.
3. Review generated SQL and execution output.

## Training Scripts

Available scripts in `training/`:

- `prepare_mixture.py`
- `train_lora.py`
- `train_mistral_qlora.py`
- `start_fast_track.ps1`

Example (PowerShell):

```powershell
.\training\start_fast_track.ps1 -BirdLocal "C:\path\to\bird" -PythonExe "python"
```

## Notes

- If adapter download fails, check internet connectivity and Google Drive file permissions.
- The project expects adapters under `training/artifacts/flan_t5_sql_lora/final` and `training/artifacts/mistral_7b_sql_lora/final`.
- PyTorch CUDA wheels are configured via `--extra-index-url` in `requirements.txt`.
