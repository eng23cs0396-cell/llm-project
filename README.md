# Enterprise Text-to-SQL Engine with NVIDIA NIM, Ollama & LangChain

An intelligent, multi-engine Text-to-SQL system that translates natural language into accurate SQL queries, executes them against databases (SQLite, DuckDB, etc.), and validates queries with syntax checkers and self-healing error reflection.

---

## 🌟 Key Features

1. **Multi-Model Inference Architecture:**
   - **NVIDIA NIM (Cloud):** High-speed LLM inference via NVIDIA NIM API (e.g. `openai/gpt-oss-20b`, `meta/llama-3.1-70b-instruct`).
   - **Local LLM (Ollama):** Private, offline inference using local Ollama (`http://localhost:11434/v1`) with models like `llama3:latest` or `mistral:7b`.
   - **Local Fine-Tuned FLAN-T5:** Compact, fast on-premise encoder-decoder fallback.
   - **Auto Mode:** Automatically routes to NVIDIA NIM if available, seamlessly failing over to local Ollama or heuristics if limits or connectivity issues occur.

2. **LangChain & Advanced Prompt Engineering:**
   - **Dynamic Few-Shot Learning:** Automatically selects the most relevant Spider and BIRD SQL patterns based on schema similarity and query complexity.
   - **Dialect Adaptation:** Tailored prompts for SQLite, PostgreSQL, MySQL, and DuckDB.
   - **Self-Healing Reflection:** When query execution fails, a reflection prompt feeds the exact database error back to the LLM to autonomously correct the SQL.

3. **Fine-Tuning Pipeline (FLAN-T5):**
   - **Parameter-Efficient Fine-Tuning (PEFT):** Supports LoRA, Rank-Stabilized LoRA (rsLoRA), and Weight-Decomposed Low-Rank Adaptation (DoRA).
   - **Google Colab Notebook:** Ready-to-run Jupyter notebook (`training/Fine_Tune_FLAN_T5.ipynb`) with GPU acceleration and dataset preparation.
   - **Production Training Script:** `training/train_flan_t5_advanced.py`.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- (Optional) NVIDIA NIM API Key from [build.nvidia.com](https://build.nvidia.com)
- (Optional) [Ollama](https://ollama.com/) installed locally for offline generation

### 2. Environment Configuration
Copy the sample environment file and configure your credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```env
# NVIDIA NIM Settings
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL_NAME=openai/gpt-oss-20b

# Local Ollama Settings (Optional)
LOCAL_MODEL_URL=http://localhost:11434/v1
LOCAL_MODEL_NAME=llama3:latest

# App Settings
PORT=8000
```

### 3. Run the Application

Using PowerShell (Windows):
```powershell
.\run_app.ps1
```

Or manually using Uvicorn:
```bash
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
- **Interactive Web UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Model Status:** [http://127.0.0.1:8000/models/status](http://127.0.0.1:8000/models/status)

---

## 📂 Project Structure

```text
├── app/
│   ├── main.py                     # FastAPI application endpoints & routing
│   ├── models.py                   # Pydantic schemas (requests, responses, model selection)
│   ├── storage.py                  # Database metadata and schema storage
│   ├── static/
│   │   └── index.html              # Modern web interface with model selector
│   └── services/
│       ├── nim_service.py          # LangChain ChatOpenAI client for NIM & Ollama
│       ├── prompt_engine.py        # Dynamic few-shot selector and dialect prompts
│       ├── retry_engine.py         # Self-healing SQL reflection loop
│       ├── schema_service.py       # Table and column schema extraction
│       ├── sql_validator.py        # SQL syntax and security validation
│       ├── sql_semantic_guard.py   # Guardrails preventing destructive operations
│       ├── retrieval_service.py    # Schema embedding and vector similarity search
│       └── llm_sql_generator.py    # Local FLAN-T5 generator and routing logic
├── training/
│   ├── Fine_Tune_FLAN_T5.ipynb     # Interactive Google Colab notebook for LoRA/DoRA
│   ├── train_flan_t5_advanced.py   # Standalone FLAN-T5 fine-tuning pipeline
│   ├── prepare_mixture.py          # Spider & BIRD dataset formatting script
│   └── data/                       # Training mixtures
├── run_app.ps1                     # PowerShell launch helper
├── requirements.txt                # Python dependencies
├── .env.example                    # Sample environment variables
└── README.md
```

---

## 🧪 Model Training & Fine-Tuning

To fine-tune FLAN-T5 on your own Text-to-SQL datasets:

1. **Via Google Colab (Recommended for Free GPUs):**
   Open `training/Fine_Tune_FLAN_T5.ipynb` in [Google Colab](https://colab.research.google.com/), select a T4 GPU runtime, and run all cells.

2. **Via Local Command Line:**
   ```bash
   python training/train_flan_t5_advanced.py \
       --train-file training/data/mixture_train.jsonl \
       --base-model google/flan-t5-base \
       --output-dir training/artifacts/flan-t5-advanced \
       --peft-type lora \
       --epochs 3
   ```
