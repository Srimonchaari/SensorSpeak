# SensorSpeak — Bosch Accelerometer Motion Event Explainer

> **Ask plain-English questions about your sensor data. Get grounded answers. No cloud. No API keys. No ML training.**

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [How It Works — The Full Idea](#3-how-it-works--the-full-idea)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Project Structure](#5-project-structure)
6. [Tech Stack](#6-tech-stack)
7. [Quickstart — Google Colab (recommended)](#7-quickstart--google-colab-recommended)
8. [Quickstart — Local Setup](#8-quickstart--local-setup)
9. [Environment Variables](#9-environment-variables)
10. [Using the Gradio Chat UI](#10-using-the-gradio-chat-ui)
11. [Uploading a Bosch PDF Datasheet](#11-uploading-a-bosch-pdf-datasheet)
12. [Connecting to OpenAI or HuggingFace](#12-connecting-to-openai-or-huggingface)
13. [Fine-Tuning with the Synthetic Dataset](#13-fine-tuning-with-the-synthetic-dataset)
14. [Tunable Parameters](#14-tunable-parameters)
15. [Running the Tests](#15-running-the-tests)
16. [Detected Event Types](#16-detected-event-types)
17. [Example Queries and Answers](#17-example-queries-and-answers)
18. [Limitations](#18-limitations)
19. [Roadmap](#19-roadmap)
20. [Contributing](#20-contributing)
21. [License](#21-license)

---

## 1. What Is This?

**SensorSpeak** is a fully local, production-ready Python pipeline that turns raw accelerometer CSV data into plain-English explanations and answers natural-language questions about motion events.

You point it at a CSV file with `timestamp`, `accel_x`, `accel_y`, `accel_z` columns — or use the built-in synthetic data generator — and you get:

- Automatically detected motion events (idle, walking, impact, shaking)
- A plain-English sentence describing each event's time, severity, and physical meaning
- A chat interface where you can ask questions like *"Were there any dangerous events?"* or *"How long was the device idle?"*
- A three-panel signal visualisation saved as a PNG
- A Gradio web UI so non-technical users can interact with the data
- Support for uploading Bosch PDF datasheets and asking questions about the sensor specs alongside your recorded data

Everything runs **on your machine**. No data leaves your device. No subscriptions. No API keys required (optional if you want richer LLM answers).

---

## 2. The Problem It Solves

### The gap between "sensor attached" and "insight delivered"

Industrial machines — motors, conveyor belts, rotating shafts, packaging lines — generate continuous accelerometer data at 100–1000 samples per second. This data is the earliest warning system for faults: a bearing starting to fail vibrates differently, an impact event might indicate a loose component, sustained shaking can signal resonance issues.

**The problem is that nobody can read raw numbers.**

A data file with 100,000 rows of floating-point accelerometer readings tells a maintenance engineer nothing actionable without:

1. Signal processing to extract meaningful features
2. Logic to classify those features into human-readable event types
3. A way to ask questions about what happened without writing code

Today, solving this requires either:
- A dedicated data science team (expensive, slow)
- A cloud analytics subscription (data privacy concerns, ongoing cost, internet dependency)
- A custom ML model trained on labelled fault data (months of effort, needs labelled ground truth)

**SensorSpeak removes all three barriers.**

It uses deterministic rule-based detection (no training data needed), a locally-running LLM (no cloud, no API key, no data leaves your machine), and a conversational interface that any engineer can use.

---

## 3. How It Works — The Full Idea

### The five-step mental model

```
Raw sensor numbers  →  Features  →  Events  →  Summaries  →  Answers
```

**Step 1 — Normalise**: Raw accelerometer values vary by sensor, mounting, and orientation. Z-score normalisation puts all three axes on the same scale so comparisons are meaningful. The raw values are preserved in `_raw_*` columns for visualisation.

**Step 2 — Features**: The resultant magnitude (`√(x²+y²+z²)`) captures total motion energy regardless of device orientation. Rolling mean and rolling std over a 0.5-second window smooth out noise and reveal patterns.

**Step 3 — Detect**: A deterministic rule engine labels every sample using four threshold conditions (see [Section 16](#16-detected-event-types)). Consecutive runs of the same label ≥ 10 samples become a `MotionEvent` object.

**Step 4 — Summarise**: Each event becomes one plain-English sentence: *"From 13.00s to 14.98s, a severe-severity 'impact' event was detected (peak magnitude 16.45 m/s²) — sharp high-amplitude spike consistent with a drop or collision."*

**Step 5 — Query**: All summaries are stored in a LlamaIndex `VectorStoreIndex`. When you ask a question, the embedding model finds the most relevant summaries, and the local Ollama LLM synthesises a grounded natural-language answer. If Ollama is not running, a keyword fallback answers from the summaries directly.

### Why rule-based detection, not a trained model?

- **No labelled data needed** — you can deploy on a brand-new sensor with zero historical fault data
- **Fully explainable** — every classification has a human-readable reason
- **Tunable in seconds** — change a constant, re-run; no retraining
- **Deterministic** — the same input always produces the same output

---

## 4. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                               │
│                                                                     │
│   Synthetic Generator  ──OR──  Real CSV Upload  ──OR──  Bosch PDF   │
│   (4 segments, 19s)            (timestamp,                          │
│                                 accel_x/y/z)                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SIGNAL PROCESSING                               │
│                                                                     │
│   Z-score normalise  →  accel_magnitude = √(x²+y²+z²)              │
│   rolling_mean (window=50)  →  rolling_std (window=50)              │
│   Raw values preserved in _raw_accel_x/y/z columns                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   RULE-BASED EVENT DETECTOR                         │
│                                                                     │
│   idle    →  rolling_std < 0.15                                     │
│   impact  →  rolling_mean > 2.5  AND  rolling_std > 1.0             │
│   shaking →  rolling_std > 1.8                                      │
│   walking →  mean ∈ [0.8, 2.5]   AND  std ∈ [0.10, 1.8]            │
│   unknown →  none of the above                                      │
│                                                                     │
│   min_event_samples = 10  (filters out noise spikes)                │
│   Output: List[MotionEvent(start, end, type, max_mag, mean_mag)]    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PLAIN-ENGLISH SUMMARISER                         │
│                                                                     │
│   "From 13.00s to 14.98s (1.98s), a severe-severity 'impact'        │
│    event was detected (peak 16.45, mean 8.23 m/s²) —               │
│    sharp high-amplitude spike; possible drop or collision."         │
└──────────────┬──────────────────────────────┬───────────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────┐    ┌─────────────────────────────────────┐
│   LLAMAINDEX VECTOR       │    │          PDF INGESTION               │
│   STORE INDEX             │    │                                     │
│                           │    │  pypdf → text chunks (1000 chars)   │
│   bge-small-en-v1.5       │◄───│  Bosch BMA400 / BMI088 datasheet    │
│   embeddings (local HF)   │    │  merged into same VectorStoreIndex  │
│   similarity_top_k = 4    │    │  → query sensor specs + events      │
└──────────────┬────────────┘    └─────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LLM QUERY ENGINE                            │
│                                                                     │
│   PRIMARY   Ollama  qwen2.5:0.5b  (local, 100% offline)             │
│   OPTIONAL  OpenAI  gpt-3.5-turbo  (set OPENAI_API_KEY)             │
│   OPTIONAL  HuggingFace Inference API  (set HF_API_KEY)             │
│   FALLBACK  Keyword matcher  (always available, no LLM needed)      │
│                                                                     │
│   response_mode = compact                                           │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INTERFACES                                        │
│                                                                     │
│   Notebook (SensorSpeak.ipynb)   →  Colab / Jupyter                │
│   Gradio UI  (ui_app.py)         →  browser chat + file upload      │
│   Python API (sensorspeak_core)  →  import and call directly        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Project Structure

```
SensorSpeak/
│
├── SensorSpeak.ipynb          ← Main Colab notebook (11 sections, run top to bottom)
│
├── sensorspeak_core.py        ← All pipeline functions (importable module)
├── llm_config.py              ← Pluggable LLM backend switcher (Ollama / OpenAI / HF)
├── ui_app.py                  ← Gradio chat UI
├── pdf_qa.py                  ← PDF upload + combined RAG index
├── finetune_prep.py           ← Fine-tuning dataset generator + instructions
│
├── tests/
│   ├── __init__.py
│   └── test_sensorspeak.py    ← 85 unit tests (pytest)
│
├── outputs/                   ← Generated at runtime, not committed to git
│   ├── synthetic_accel_data.csv
│   ├── detected_events.csv
│   ├── accelerometer_overview.png
│   └── finetune_dataset.jsonl
│
├── README.md                  ← This file
├── QUICKSTART.md              ← One-page setup + troubleshooting table
├── CLAUDE.md                  ← Claude Code session context (project rules)
├── agent.md                   ← Autonomous agent behaviour rules
├── requirements.txt           ← Pinned Python dependencies
└── .gitignore
```

---

## 6. Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| pandas | ≥ 2.0.0 | DataFrame operations, CSV I/O |
| numpy | ≥ 1.24.0 | Signal processing, array math |
| matplotlib | ≥ 3.7.0 | 3-panel sensor visualisation |
| llama-index-core | ≥ 0.10.0 | Vector index, query engine framework |
| llama-index-llms-ollama | ≥ 0.1.0 | Ollama ↔ LlamaIndex bridge |
| llama-index-embeddings-huggingface | ≥ 0.1.0 | HuggingFace embedding integration |
| sentence-transformers | ≥ 2.2.0 | Loads bge-small-en-v1.5 locally |
| Ollama | latest | Local LLM server daemon |
| qwen2.5:0.5b | 0.5B params | Default inference LLM (~350 MB) |
| BAAI/bge-small-en-v1.5 | — | Sentence embedding model |
| Gradio | ≥ 4.0.0 | Web chat UI |
| pypdf | ≥ 3.0.0 | PDF text extraction |
| Pillow | ≥ 10.0.0 | Image handling in Gradio |
| pytest | ≥ 7.4.0 | Unit test runner |

---

## 7. Quickstart — Google Colab (recommended)

No local installation needed. Everything runs in the browser.

**Step 1 — Open the notebook**

Upload `SensorSpeak.ipynb` to [colab.research.google.com](https://colab.research.google.com/) via **File → Open notebook → Upload tab**.

**Step 2 — Set runtime to GPU**

Go to **Runtime → Change runtime type → T4 GPU**.
This is free and speeds up the HuggingFace embedding model (~3× faster).

**Step 3 — Install Ollama inside Colab**

In **Section 6** of the notebook, find the line:
```python
# install_ollama_colab()
```
Uncomment it (remove the `#`) and run the cell. This downloads and starts Ollama (~2 min).

**Step 4 — Run all cells**

**Runtime → Run all** (`Ctrl+F9`).
The full pipeline takes about 5–8 minutes on first run (model download + embedding generation).

**Step 5 — Ask questions**

Scroll to **Section 11**. Edit `MY_QUESTION`:
```python
MY_QUESTION = 'Were there any dangerous events?'
```
Run the cell. You will see a grounded answer citing the specific events.

**Step 6 — Use the chat UI (optional)**

In a new Colab cell, run:
```python
!python ui_app.py --share &
import time; time.sleep(4)
```
A public Gradio URL (e.g. `https://abc123.gradio.live`) will appear. Open it in any browser.

---

## 8. Quickstart — Local Setup

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running

```bash
# 1. Clone the repository
git clone https://github.com/yourname/SensorSpeak.git
cd SensorSpeak

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the default LLM (one-time, ~350 MB)
ollama pull qwen2.5:0.5b

# 5. Start the Ollama daemon (keep this terminal open)
ollama serve

# 6a. Run as a Jupyter notebook
jupyter notebook SensorSpeak.ipynb

# 6b. OR run the Gradio chat UI
python ui_app.py
# Open http://localhost:7860 in your browser
```

---

## 9. Environment Variables

No environment variables are required for the default Ollama setup.
Variables are only needed when switching to cloud LLM backends.

| Variable | Required for | How to obtain | Example |
|---|---|---|---|
| `OPENAI_API_KEY` | OpenAI backend | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `sk-proj-...` |
| `HF_API_KEY` | HuggingFace API backend | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free) | `hf_...` |
| `SENSOSPEAK_BACKEND` | Backend selection | Set to `ollama`, `openai`, `hf_api`, or `hf_local` | `openai` |
| `OLLAMA_HOST` | Non-default Ollama port | Only if Ollama runs on a different host/port | `http://localhost:11434` |

### Setting environment variables

**macOS / Linux (terminal)**
```bash
export OPENAI_API_KEY=sk-proj-your-key-here
export SENSOSPEAK_BACKEND=openai
python ui_app.py
```

**Windows (Command Prompt)**
```cmd
set OPENAI_API_KEY=sk-proj-your-key-here
set SENSOSPEAK_BACKEND=openai
python ui_app.py
```

**Google Colab (notebook cell)**
```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-proj-your-key-here'
os.environ['SENSOSPEAK_BACKEND'] = 'openai'
```

**Permanent (macOS/Linux — add to `~/.zshrc` or `~/.bashrc`)**
```bash
echo 'export OPENAI_API_KEY=sk-proj-your-key-here' >> ~/.zshrc
source ~/.zshrc
```

---

## 10. Using the Gradio Chat UI

The Gradio UI (`ui_app.py`) gives any user — technical or not — a browser-based interface to the full pipeline.

### Starting the UI

```bash
# Local (opens at http://localhost:7860)
python ui_app.py

# With a public share link (useful for demos and Colab)
python ui_app.py --share
```

### UI Layout

```
┌─────────────────────────┬──────────────────────────────────────────┐
│  ⚙️ Configuration        │  💬 Ask Questions About Your Sensor Data │
│                         │                                          │
│  [Upload CSV]           │  ┌─────────────────────────────────────┐ │
│                         │  │  SensorSpeak Chat                   │ │
│  LLM Backend            │  │                                     │ │
│  [Dropdown ▼]           │  │  You: Were there any impacts?       │ │
│                         │  │  Bot: A severe-severity impact ...  │ │
│  [▶ Run Pipeline]       │  │                                     │ │
│                         │  └─────────────────────────────────────┘ │
│  Pipeline Status        │                                          │
│  ✅ Pipeline ready      │  [Your question...         ] [Send]      │
│  Samples: 1,900         │                                          │
│  Events: 4              │  [Clear chat]                            │
│  Backend: ollama        │                                          │
│                         │  📊 Signal Overview                      │
│  📋 Detected Events     │  ┌─────────────────────────────────────┐ │
│  | # | Type | Start ... │  │  [3-panel matplotlib figure]        │ │
│  | 1 | idle | 0.00  ... │  │                                     │ │
│  | 2 | walk | 5.00  ... │  └─────────────────────────────────────┘ │
└─────────────────────────┴──────────────────────────────────────────┘
```

### Step-by-step walkthrough

**1. Upload a CSV (optional)**
Click **Upload Sensor CSV**. Your file must have these columns:
```
timestamp, accel_x, accel_y, accel_z
```
Leave empty to use the built-in 19-second synthetic dataset.

**2. Choose a backend**
Select from the dropdown:
- `Ollama (local, no key)` — default, fully offline, needs Ollama running
- `OpenAI (needs OPENAI_API_KEY)` — richer answers, costs money per query
- `HuggingFace API (needs HF_API_KEY)` — free tier available
- `HuggingFace Local (no key)` — downloads model on first use, then offline

**3. Click Run Pipeline**
The status panel shows:
```
✅ Pipeline ready
  Samples : 1,900
  Events  : 4
  Backend : ollama  (LlamaIndex + LLM)
  Source  : synthetic data
```

**4. Ask questions in the chat**
Type any natural-language question and press **Send** or **Enter**:
```
Were there any severe or high-severity events?
How long was the device idle?
What happened between 13 and 15 seconds?
Describe the shaking event.
What was the peak acceleration magnitude?
Were there any walking or rhythmic patterns?
Summarise all events in order.
```

**5. View the signal plot**
The three-panel figure at the bottom updates after each pipeline run. The coloured bands on Panel 2 correspond to detected events.

---

## 11. Uploading a Bosch PDF Datasheet

SensorSpeak can ingest Bosch sensor datasheets (BMA400, BMI088, BMG250, etc.) and answer questions about sensor specifications alongside your recorded data — all in the same chat.

### Where to download Bosch datasheets

Bosch publishes datasheets for all their MEMS sensors at:
- [https://www.bosch-sensortec.com/products/motion-sensors/accelerometers/](https://www.bosch-sensortec.com/products/motion-sensors/accelerometers/)

Download the PDF for your sensor (e.g. BMA400 Datasheet v1.5).

### Method 1 — Python API (notebook or script)

```python
from pdf_qa import build_combined_index, query_combined

# Point to your downloaded PDFs
index = build_combined_index(
    pdf_paths=['BMA400_datasheet.pdf', 'BMI088_datasheet.pdf'],
    events=events,       # from detect_events()
    summaries=summaries, # from summarize_event()
)

# Query specs and events together
print(query_combined(index, 'What is the measurement range of the BMA400?'))
print(query_combined(index, 'Were there any impact events that might have exceeded sensor limits?'))
print(query_combined(index, 'What is the operating voltage and noise density?'))
```

### Method 2 — Google Colab interactive upload

Add this cell in the notebook after Section 9:

```python
from pdf_qa import upload_and_build, query_combined

# Opens a file picker in Colab
combined_index = upload_and_build(events, summaries)

# Ask questions spanning both the datasheet and your sensor data
answer = query_combined(combined_index, 'What is the full-scale range for acceleration?')
print(answer)
```

### Method 3 — Gradio UI

In the UI, after clicking **Run Pipeline**, you can extend the index by running this in a notebook cell alongside the UI:

```python
from pdf_qa import build_combined_index
from sensorspeak_core import run_pipeline

result = run_pipeline()
combined_index = build_combined_index(
    pdf_paths=['BMA400_datasheet.pdf'],
    events=result['events'],
    summaries=result['summaries'],
)
```

### How PDF chunking works

The PDF text is split into 1000-character overlapping chunks (100-char overlap). Each chunk becomes a LlamaIndex `Document` with metadata:
```python
{
  'source': 'BMA400_datasheet.pdf',
  'chunk': 3,
  'total_chunks': 87,
  'doc_type': 'datasheet'
}
```
This means the LLM can answer questions like *"On which page is the noise density spec?"* by citing the source document and chunk.

### Troubleshooting PDF upload

| Problem | Cause | Fix |
|---|---|---|
| `ImportError: pypdf` | pypdf not installed | `pip install pypdf` |
| `No extractable text found` | PDF is scanned (image-only) | Run OCR first: `pip install ocrmypdf && ocrmypdf input.pdf output.pdf` |
| Very long indexing time | Large PDF (100+ pages) | Increase `chunk_size=2000` to reduce document count |
| Irrelevant answers | Chunks too small, context split | Increase `chunk_size=1500, overlap=200` |

---

## 12. Connecting to OpenAI or HuggingFace

By default SensorSpeak uses Ollama locally. To switch backends, set one environment variable and optionally provide an API key.

### OpenAI (GPT-3.5 / GPT-4)

**When to use**: You want longer, more detailed answers and don't mind per-query cost (~$0.001 per question with GPT-3.5).

```bash
# 1. Set your API key
export OPENAI_API_KEY=sk-proj-your-key-here

# 2. Switch the backend
export SENSOSPEAK_BACKEND=openai

# 3. Run
python ui_app.py
```

Or in Python directly:

```python
from llm_config import get_llm, get_embed_model, LLMBackend
from sensorspeak_core import run_pipeline

llm         = get_llm(LLMBackend.OPENAI)           # uses gpt-3.5-turbo by default
embed_model = get_embed_model(LLMBackend.OPENAI)   # uses text-embedding-3-small

result = run_pipeline(llm=llm, embed_model=embed_model)
```

To use GPT-4:
```python
llm = get_llm(LLMBackend.OPENAI, model='gpt-4o-mini')
```

Install the OpenAI integration:
```bash
pip install llama-index-llms-openai llama-index-embeddings-openai
```

---

### HuggingFace Inference API (free tier)

**When to use**: You want a cloud LLM without paying for OpenAI. The free tier handles ~1000 requests/day.

```bash
# 1. Get a free token at huggingface.co/settings/tokens
export HF_API_KEY=hf_your_token_here

# 2. Switch backend
export SENSOSPEAK_BACKEND=hf_api

# 3. Run
python ui_app.py
```

Or in Python:
```python
from llm_config import get_llm, LLMBackend

llm = get_llm(LLMBackend.HUGGINGFACE_API)
# Default model: HuggingFaceH4/zephyr-7b-beta

# Use a different model:
llm = get_llm(LLMBackend.HUGGINGFACE_API, model='mistralai/Mistral-7B-Instruct-v0.3')
```

Install:
```bash
pip install llama-index-llms-huggingface-api
```

---

### HuggingFace Local (no key, no Ollama)

**When to use**: You want a local model but don't want to install Ollama. The model downloads on first run and is cached.

```bash
export SENSOSPEAK_BACKEND=hf_local
python ui_app.py
```

Or in Python:
```python
from llm_config import get_llm, LLMBackend

llm = get_llm(LLMBackend.HUGGINGFACE_LOCAL, model='HuggingFaceH4/zephyr-7b-beta')
```

Install:
```bash
pip install llama-index-llms-huggingface transformers accelerate
```

---

### Backend comparison

| Backend | Env var needed | Internet | Cost | Answer quality | Speed |
|---|---|---|---|---|---|
| **Ollama** (default) | None | No | Free | Good | Fast on M-series / GPU |
| **OpenAI** | `OPENAI_API_KEY` | Yes | ~$0.001/query | Best | Fast (API) |
| **HuggingFace API** | `HF_API_KEY` | Yes | Free tier | Good | Moderate |
| **HuggingFace Local** | None | First run only | Free | Good | Slow on CPU |
| **Keyword fallback** | None | No | Free | Basic | Instant |

To check your current backend configuration at any time:
```python
from llm_config import describe_backend
print(describe_backend())
```

---

## 13. Fine-Tuning with the Synthetic Dataset

Fine-tuning teaches a small model sensor-domain vocabulary so it gives more precise answers about event types, threshold rules, and physical interpretations.

### Step 1 — Generate the fine-tuning dataset

```bash
python finetune_prep.py
# Writes: outputs/finetune_dataset.jsonl  (200+ instruction-response pairs)
```

The dataset contains three types of samples:
- **Classification pairs** — "Classify this reading: rolling_mean=1.4, rolling_std=0.6" → explains the rule that fired
- **Event Q&A** — questions about specific time windows → answers from event summaries
- **Threshold Q&A** — questions about detection rules, gravity offset, sensor behaviour

Each sample is in Alpaca format:
```json
{
  "instruction": "Classify this accelerometer reading and explain why:\nrolling_mean = 1.423, rolling_std = 0.612",
  "input": "",
  "output": "This reading is classified as \"walking\" (low severity).\nReason: rolling_mean (1.423) is in [0.8, 2.5] and rolling_std (0.612) is in [0.10, 1.8], consistent with rhythmic periodic motion such as footsteps."
}
```

---

### Option A — HuggingFace PEFT + LoRA (Colab T4, ~30 minutes)

Best for: permanent improvement, exportable model, no GPU required after training.

```bash
# Install training dependencies
pip install peft trl datasets transformers accelerate bitsandbytes
```

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'   # matches our default Ollama model

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map='auto',
)

lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules='all-linear',
    lora_dropout=0.05,
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_cfg)

dataset = load_dataset('json', data_files='outputs/finetune_dataset.jsonl', split='train')

def format_sample(ex):
    return {'text': f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"}

dataset = dataset.map(format_sample)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir='./sensospeak_lora',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        logging_steps=10,
        dataset_text_field='text',
    ),
)
trainer.train()
model.save_pretrained('./sensospeak_lora')
print('Fine-tuning complete. Model saved to ./sensospeak_lora')
```

**Use the fine-tuned model in SensorSpeak:**
```python
from llm_config import get_llm, LLMBackend
llm = get_llm(LLMBackend.HUGGINGFACE_LOCAL, model='./sensospeak_lora')
result = run_pipeline(llm=llm)
```

---

### Option B — Ollama Modelfile (no training, no GPU, 5 minutes)

Best for: quick improvement to Ollama answers without any training at all.

Create a file named `Modelfile` in the project root:

```
FROM qwen2.5:0.5b

SYSTEM """
You are SensorSpeak, an expert at interpreting Bosch MEMS accelerometer data.

You understand these motion event types and their detection rules:
- idle:    rolling_std < 0.15  →  device at rest, minimal vibration
- walking: rolling_mean 0.8–2.5 AND rolling_std 0.10–1.8  →  rhythmic periodic motion
- impact:  rolling_mean > 2.5 AND rolling_std > 1.0  →  drop, collision, or hammer blow
- shaking: rolling_std > 1.8  →  rapid sustained oscillation

Always cite the time range, peak magnitude, severity level, and physical explanation in your answers.
Severity: low < 1.5 m/s²  |  moderate 1.5–3.5  |  high 3.5–7.0  |  severe > 7.0
"""
```

Register and use it:
```bash
ollama create sensospeak -f Modelfile
ollama run sensospeak   # test it interactively
```

Update the model in `sensorspeak_core.py`:
```python
OLLAMA_MODEL = 'sensospeak'
```

---

### Option C — HuggingFace AutoTrain (no-code, web browser)

Best for: non-coders who want to fine-tune without writing any Python.

1. Go to [https://ui.autotrain.huggingface.co](https://ui.autotrain.huggingface.co)
2. Create a new project → **Text Generation** (Causal LM)
3. Upload `outputs/finetune_dataset.jsonl`
4. Set **instruction column** = `instruction`, **response column** = `output`
5. Select model: `Qwen/Qwen2.5-0.5B-Instruct`
6. Click **Train** — HuggingFace runs the job in their cloud (~20 min)
7. Download the model and use with `LLMBackend.HUGGINGFACE_LOCAL`

---

## 14. Tunable Parameters

All parameters are named constants defined at the top of their respective cells in `SensorSpeak.ipynb` and mirrored in `sensorspeak_core.py`. Change the value and re-run from that cell — no code changes needed.

### Data Generation

| Parameter | Default | Description | Tune when... |
|---|---|---|---|
| `SAMPLE_RATE_HZ` | `100` | Samples per second | Your real sensor runs at a different rate |
| `GRAVITY_Z` | `9.81` | Gravity offset on Z axis (m/s²) | Device mounted at an angle |
| `SEG_IDLE_DURATION` | `5` | Idle segment length (s) | You want more/less idle baseline |
| `SEG_WALK_DURATION` | `8` | Walking segment length (s) | Adjusting dataset balance |
| `SEG_IMPACT_DURATION` | `2` | Impact segment length (s) | Adjusting dataset balance |
| `SEG_SHAKE_DURATION` | `4` | Shaking segment length (s) | Adjusting dataset balance |
| `IMPACT_SPIKE` | `12.0` | Peak spike magnitude (m/s²) | Simulating lighter or heavier impacts |
| `SHAKE_AMPLITUDE` | `4.0` | Shaking oscillation amplitude | Simulating stronger/weaker agitation |
| `RANDOM_SEED` | `42` | NumPy RNG seed | Generating varied synthetic datasets |

### Signal Processing

| Parameter | Default | Description | Tune when... |
|---|---|---|---|
| `ROLLING_WINDOW` | `50` | Rolling stats window (samples = 0.5s) | Higher sample rate or noisier sensor |
| `ZSCORE_EPS` | `1e-8` | Division-by-zero guard in z-score | Never needs changing |

### Event Detection

| Parameter | Default | Description | Tune when... |
|---|---|---|---|
| `IDLE_STD_MAX` | `0.15` | Max std for idle classification | Sensor is noisy at rest → increase |
| `IMPACT_MEAN_MIN` | `2.5` | Min mean for impact | Missing light impacts → decrease |
| `IMPACT_STD_MIN` | `1.0` | Min std for impact | Too many false positives → increase |
| `SHAKING_STD_MIN` | `1.8` | Min std for shaking | Walking misclassified as shaking → increase |
| `WALKING_MEAN_MIN` | `0.8` | Lower bound mean for walking | Slow gait → decrease |
| `WALKING_MEAN_MAX` | `2.5` | Upper bound mean for walking | Fast gait classified as impact → increase |
| `WALKING_STD_MIN` | `0.10` | Lower bound std for walking | Very smooth gait → decrease |
| `WALKING_STD_MAX` | `1.8` | Upper bound std for walking | Overlap with shaking → decrease |
| `MIN_EVENT_SAMPLES` | `10` | Minimum run length to register event | Noisy signal, too many short events → increase |

### LLM Settings

| Parameter | Default | Description | Tune when... |
|---|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Ollama model tag | Want richer answers → swap to `qwen2.5:3b` |
| `OLLAMA_TIMEOUT` | `180` | Query timeout (seconds) | Slow machine → increase |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model | Need better retrieval → try `bge-base-en-v1.5` |
| `SIMILARITY_TOP_K` | `4` | Documents retrieved per query | More context → increase; faster → decrease |
| `RESPONSE_MODE` | `compact` | LlamaIndex synthesis mode | Longer answers → try `tree_summarize` |

---

## 15. Running the Tests

The test suite covers all pipeline functions with 85 tests across 10 test classes.

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run a single test class
pytest tests/test_sensorspeak.py::TestDetectEvents -v

# Run with coverage report
pip install pytest-cov
pytest tests/ --cov=sensorspeak_core --cov-report=term-missing
```

Expected output:
```
============================= test session starts ==============================
collected 85 items

tests/test_sensorspeak.py::TestGenerateSyntheticData::test_returns_dataframe PASSED
tests/test_sensorspeak.py::TestGenerateSyntheticData::test_expected_row_count PASSED
...
============================== 85 passed in 5.86s ==============================
```

### What is tested

| Class | Tests | What is verified |
|---|---|---|
| `TestGenerateSyntheticData` | 13 | Shape, timestamps, gravity offset, spike injection, reproducibility |
| `TestNormalizeAndEngineerFeatures` | 13 | Z-score properties, new columns, error handling, immutability |
| `TestClassifySample` | 10 | All five rule branches, boundary conditions |
| `TestDetectEvents` | 11 | Event structure, min-samples guard, empty input, missing columns |
| `TestSeverityLabel` | 7 | All four tiers and all three exact boundaries |
| `TestSummarizeEvent` | 9 | Content, structure, severity word, all events covered |
| `TestKeywordFallback` | 7 | Relevance scoring, top-k limit, edge cases |
| `TestQueryEvents` | 4 | Empty questions, fallback without index |
| `TestRunPipeline` | 6 | End-to-end, bad CSV detection, real CSV roundtrip |
| `TestMotionEvent` | 2 | Field access, duration computation |

---

## 16. Detected Event Types

| Event | Detection Rule | Physical Meaning | Real-World Example |
|---|---|---|---|
| `idle` | rolling_std < 0.15 | Device stationary; no significant motion | Machine off, device on desk, powered-down conveyor |
| `walking` | mean ∈ [0.8, 2.5] AND std ∈ [0.10, 1.8] | Rhythmic periodic motion | Footsteps, rotating shaft, belt conveyor at normal speed |
| `impact` | mean > 2.5 AND std > 1.0 | Sudden high-amplitude transient | Drop, collision, hammer blow, tooling strike |
| `shaking` | std > 1.8 | Rapid sustained oscillation | Intentional agitation, loose component vibrating, bearing fault |
| `unknown` | none of the above | Unclassified motion | Transition between states; thresholds may need tuning |

---

## 17. Example Queries and Answers

**Q: Were there any abnormal or high-severity events?**
> Two high-severity events were detected. A severe-severity 'impact' event occurred from 13.00s to 14.98s (peak magnitude 16.45 m/s²) — a sharp high-amplitude spike consistent with a drop or collision. A 'shaking' event of high severity was recorded from 15.00s to 18.99s (peak magnitude 6.82 m/s²) — rapid high-frequency oscillation indicating intentional agitation.

**Q: How long did the device remain idle?**
> The device was idle from 0.00s to 4.99s, a total of approximately 5 seconds. During this period the signal showed minimal vibration with a mean magnitude of 0.04 m/s², consistent with a stationary device at rest.

**Q: What is the operating voltage of the BMA400?** *(when a Bosch datasheet is loaded)*
> According to the BMA400 datasheet (Page 12), the digital supply voltage VDD IO ranges from 1.2V to 3.6V, and the core supply voltage ranges from 1.71V to 3.6V. The device draws approximately 14.5 μA in normal power mode.

---

## 18. Limitations

- **Thresholds need per-sensor tuning** — the default values are calibrated for the synthetic dataset. Real sensors will need 15–30 minutes of calibration runs to set correct values.
- **Synthetic data only in the notebook** — the CSV upload cell exists but is commented out. Real hardware integration (Bosch BHI260, BMA400 via USB/MQTT) is planned for Phase 2.
- **0.5B model gives short answers** — the default `qwen2.5:0.5b` is chosen for speed and offline use. Switch to `qwen2.5:3b` or OpenAI for longer, more detailed explanations.
- **Single-file notebook architecture** — not yet packaged as an importable library with proper versioning (planned for Phase 2).
- **No temporal context between queries** — each question is answered independently; the LLM does not remember previous questions in the same session.
- **PDF OCR not built-in** — scanned PDFs (image-only, no text layer) will produce no extractable text. Pre-process with `ocrmypdf` before ingestion.
- **English only** — all summaries, seeds, and LLM prompts are in English.

---

## 19. Roadmap

**Phase 1 — Offline Notebook (complete)**
Full LlamaIndex + Ollama pipeline in Google Colab. Synthetic data generator, rule-based detection, vector index, NL query interface, Gradio UI, PDF ingestion, fine-tuning dataset, 85-test suite.

**Phase 2 — Live Hardware + API Endpoint**
Streaming ingestion from Bosch BMA400/BMI088/BHI260 via serial (pyserial) or MQTT (paho-mqtt). FastAPI `/query` and `/events` endpoints. Docker container for edge deployment. Real-time threshold auto-calibration from a 30-second baseline recording.

**Phase 3 — Streamlit Dashboard + Alerts + Reports**
Real-time Streamlit UI with live event feed and rolling statistics charts. Threshold-breach alerts via email/Slack webhook. Per-shift PDF report generation with event timeline and LLM summaries. Multi-sensor comparison dashboard.

---

## 20. Contributing

Contributions are welcome. Before opening a pull request:

1. Fork the repository and create a feature branch
2. Run the test suite — all 85 tests must pass: `pytest tests/ -v`
3. For new event types: update `_classify_sample()`, `_SEED_MAP`, `EVENT_COLORS`, the README event table, and `CLAUDE.md`
4. For new constants: add to the named-constants block at the top of the relevant cell/module and update the README parameter table
5. Open issues for:
   - Threshold suggestions tuned to specific Bosch sensor models
   - New event type proposals with detection rule and physical justification
   - Hardware integration examples with specific sensor/board combinations

---

## 21. License

MIT License — free to use, modify, and distribute with attribution.

---

## Author

[Your Name] · [LinkedIn URL] · [GitHub URL]

---

*Built with Python, LlamaIndex, Ollama, and a Bosch accelerometer dataset.*
*No cloud. No API keys. No ML training. Just signal processing and local LLMs.*
