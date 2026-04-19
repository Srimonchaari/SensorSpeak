# SensorSpeak — Bosch Accelerometer Motion Event Explainer

A fully local Python pipeline that reads raw accelerometer data, detects motion events, and answers plain-English questions about what happened.

No cloud. No API keys required. No ML training.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What is an Accelerometer?](#2-what-is-an-accelerometer)
3. [What Readings Mean](#3-what-readings-mean)
4. [Problem](#4-problem)
5. [Approach](#5-approach)
6. [Components Used](#6-components-used)
7. [Models / LLM Layer Used](#7-models--llm-layer-used)
8. [Why LangChain and LangGraph](#8-why-langchain-and-langgraph)
9. [Pipeline](#9-pipeline)
10. [Event Types](#10-event-types)
11. [Example Input](#11-example-input)
12. [Example Output](#12-example-output)
13. [V1 Scope](#13-v1-scope)
14. [V2 Ideas](#14-v2-ideas)
15. [Setup](#15-setup)
16. [Project Structure](#16-project-structure)
17. [Tunable Parameters](#17-tunable-parameters)
18. [Running the Tests](#18-running-the-tests)
19. [License](#19-license)

---

## 1. Project Overview

**SensorSpeak** takes a CSV file from a Bosch accelerometer — or generates synthetic data automatically — and does three things:

1. **Detects motion events** in the signal using rule-based logic. No machine learning, no training data.
2. **Writes a plain-English summary** of each event: when it happened, how long it lasted, how severe it was, and what it likely means physically.
3. **Answers natural-language questions** about those events using a local LLM orchestrated by LangChain and LangGraph.

You can ask things like:

- "Did any impact occur?"
- "When was the device idle?"
- "Was there any shaking, and how severe was it?"
- "What was the peak acceleration?"

The LLM reads the pre-written event summaries, finds the relevant ones, and writes a grounded answer. It can only answer from what the sensor actually recorded — it does not make things up.

---

## 2. What is an Accelerometer?

An accelerometer is a sensor that measures how fast something is accelerating in three directions at once.

The Bosch MEMS accelerometers (BMA400, BMI088, BMG250, etc.) are small chips used in phones, fitness trackers, industrial machines, conveyor belts, and robots. They output a continuous stream of numbers at a fixed rate — typically 100 readings per second.

Those numbers tell you whether the device is:
- Sitting still (barely any change)
- Being carried or walked with (rhythmic oscillation)
- Dropped or hit (sudden large spike)
- Shaken rapidly (high-frequency oscillation)

Raw accelerometer readings by themselves are just numbers. SensorSpeak turns those numbers into words.

---

## 3. What Readings Mean

Every row in the CSV has four columns:

| Column | What it measures | Unit |
|---|---|---|
| `timestamp` | Time since recording started | seconds |
| `accel_x` | Lateral acceleration — left/right movement | m/s² |
| `accel_y` | Forward acceleration — front/back movement | m/s² |
| `accel_z` | Vertical acceleration — up/down movement, plus gravity | m/s² |

**About the gravity offset on Z:** When the device sits flat and still on a table, `accel_z` reads approximately `+9.81 m/s²` — that is gravity pulling the sensor downward. The sensor is always "feeling" gravity, so a still device does not read zero on Z. This is normal and expected.

A completely still device produces approximately:
```
accel_x ≈ 0.00
accel_y ≈ 0.00
accel_z ≈ 9.81
```

When the device is picked up and shaken, all three values change rapidly. The pipeline combines all three into a single number called **magnitude** (`√(x²+y²+z²)`), which represents total motion energy regardless of which direction things moved.

---

## 4. Problem

A raw accelerometer CSV might have 100,000 rows of floating-point numbers. Those numbers tell a maintenance engineer nothing actionable without:

1. Signal processing to smooth noise and extract features
2. Logic to classify those features into human-readable event types
3. A way to ask questions about what happened without writing code

Today, solving this requires either a data science team, a cloud analytics subscription, or a custom ML model trained on labelled fault data. All three are expensive, slow, or have data-privacy concerns.

---

## 5. Approach

SensorSpeak solves this in layers:

**Layer 1 — Signal processing.** Normalise the three axes, compute magnitude, and smooth with rolling statistics. This removes noise and reveals patterns.

**Layer 2 — Rule-based detection.** Compare the smoothed rolling mean and standard deviation against fixed thresholds. Consecutive runs of the same label become a `MotionEvent` object. No training data is needed — if you can describe what "shaking" looks like numerically, you can detect it.

**Layer 3 — Plain-English summaries.** Each detected event becomes one sentence: its time range, duration, severity, peak value, and a physical interpretation.

**Layer 4 — LangChain + LangGraph.** The summaries become documents in a vector store. A LangGraph workflow handles the full query pipeline — from receiving a user question to retrieving relevant summaries to generating an LLM answer. If the LLM is unavailable, a keyword fallback answers directly from the summaries.

---

## 6. Components Used

| File | What it does |
|---|---|
| `sensorspeak_core.py` | All pipeline functions: data generation, normalisation, feature engineering, IQR outlier removal, rule-based event detection, summarisation, vector index building, NL querying |
| `llm_config.py` | Pluggable LLM backend — swap between Ollama, OpenAI, HuggingFace API, or HuggingFace local with one setting |
| `api.py` | FastAPI REST server — exposes `/api/run`, `/api/chat`, `/api/health`; serves the React frontend from `frontend/dist` |
| `frontend/` | React web UI (Vite + TypeScript + Tailwind CSS + Framer Motion) — drag-drop CSV upload, pipeline status terminal, event list, chat interface, interactive signal chart |
| `SensorSpeak.ipynb` | Full pipeline as a Google Colab notebook — run top to bottom |
| `pdf_qa.py` | Bosch sensor datasheet ingestion — ask questions about sensor specs alongside recorded data |
| `finetune_prep.py` | Generates a fine-tuning dataset (200+ instruction-response pairs in Alpaca format) for improving LLM domain knowledge |
| `tests/` | 85 pytest unit tests covering every pipeline function |

---

## 7. Models / LLM Layer Used

| Component | Model | Where it runs |
|---|---|---|
| LLM — default | `qwen2.5:0.5b` via Ollama | Your local machine |
| LLM — optional | `gpt-3.5-turbo` or `gpt-4` via OpenAI | Cloud (needs `OPENAI_API_KEY`) |
| LLM — optional | `zephyr-7b-beta` via HuggingFace Inference API | Cloud (needs `HF_API_KEY`, free tier) |
| LLM — optional | Any HuggingFace causal LM | Your local machine |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers | Your local machine |

**The LLM is used only to synthesise answers.** Event classification is done entirely by the rule engine before the LLM is ever called. If Ollama is not running, the system falls back to keyword matching and still answers from the pre-written summaries.

The embedding model (`bge-small-en-v1.5`) converts event summary text into vectors so that the retrieval step can find the summaries most relevant to a given user question.

---

## 8. Why LangChain and LangGraph

### LangChain

LangChain provides the building blocks for connecting LLMs to data:

- **Document loaders** — turn event summaries into `Document` objects with metadata (event type, time range, magnitude)
- **Embedding integrations** — unified interface to HuggingFace, OpenAI, and other embedding providers
- **Vector stores** — store and retrieve embedded documents by semantic similarity
- **LLM interfaces** — call Ollama, OpenAI, or HuggingFace through a single consistent API

LangChain means you can swap the LLM from Ollama to OpenAI by changing one line, without touching any other code.

### LangGraph

LangGraph provides a stateful workflow layer on top of LangChain:

- Defines the query pipeline as a directed graph of steps (nodes) with explicit transitions (edges)
- Holds state across steps — retrieved documents are available when the LLM generates its answer
- Makes it easy to add branching: if Ollama is unavailable, take the keyword fallback path instead
- Keeps each step independently testable and traceable

Without LangGraph, the pipeline is a linear script. With LangGraph, the pipeline has explicit state, observable transitions, and clear points to extend or branch.

---

## 9. Pipeline

```
Raw accelerometer CSV (timestamp, accel_x, accel_y, accel_z)
         │
         ▼
┌────────────────────────────────────────┐
│  Signal Processing                     │
│  - Z-score normalise x, y, z axes      │
│  - accel_magnitude = √(x² + y² + z²)  │
│  - rolling_mean over 50-sample window  │
│  - rolling_std  over 50-sample window  │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│  IQR Outlier Removal                   │
│  - Winsorise magnitude at lower fence  │
│  - Preserves high-energy events intact │
│  - Recomputes rolling stats on clean   │
│    magnitude                           │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│  Rule-Based Event Detector             │
│                                        │
│  idle    → rolling_std < 0.15          │
│  impact  → mean > 2.5  AND std > 1.0   │
│  shaking → std > 1.8                   │
│  walking → mean ∈ [0.8, 2.5]           │
│             AND std ∈ [0.10, 1.8]      │
│  unknown → none of the above           │
│                                        │
│  Runs of ≥ 10 samples → MotionEvent    │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│  Event Summariser                      │
│  One plain-English sentence per event  │
│  (time, duration, severity, peak,      │
│   physical interpretation)             │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│  LangChain                             │
│  - Load summaries as Documents         │
│  - Embed with bge-small-en-v1.5        │
│  - Store in vector index               │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│  LangGraph Workflow                    │
│  Node 1 — receive user question        │
│  Node 2 — retrieve top-k documents     │
│  Node 3 — generate LLM answer          │
│  Node 4 — return answer to user        │
│  Fallback edge — keyword match if no   │
│                  LLM available         │
└────────────────────┬───────────────────┘
                     │
                     ▼
             Plain-English Answer
```

---

## 10. Event Types

| Event | Detection Rule | Physical Meaning | Real Example |
|---|---|---|---|
| `idle` | rolling_std < 0.15 | Device stationary, no significant motion | Machine off, device on a desk, powered-down conveyor |
| `walking` | mean ∈ [0.8, 2.5] and std ∈ [0.10, 1.8] | Rhythmic periodic motion | Person walking, rotating shaft, belt conveyor |
| `impact` | mean > 2.5 and std > 1.0 | Sudden high-energy transient | Device dropped, collision, tool strike |
| `shaking` | std > 1.8 | Rapid sustained oscillation | Intentional agitation, loose component, bearing fault |
| `unknown` | none of the above | Unclassified | State transition; thresholds may need tuning |

Severity is determined from peak magnitude:

| Severity | Peak magnitude |
|---|---|
| low | < 1.5 m/s² |
| moderate | 1.5 – 3.5 m/s² |
| high | 3.5 – 7.0 m/s² |
| severe | > 7.0 m/s² |

---

## 11. Example Input

A CSV file with four columns, sampled at 100 Hz (one row per 10 milliseconds):

```csv
timestamp,accel_x,accel_y,accel_z
0.00,0.02,-0.01,9.82
0.01,0.03,0.00,9.80
0.02,-0.01,0.02,9.83
...
13.00,8.42,3.17,15.63
13.01,6.21,2.88,14.22
...
```

The row at 13.00s has large values across all axes — that is the impact event.

Leave the file upload empty to use the built-in synthetic dataset: 4 segments (idle → walking → impact → shaking), 19 seconds total, 1,900 rows.

---

## 12. Example Output

**Detected events after running the pipeline:**

```
Event   Start    End      Duration  Severity  Peak
idle    0.00s    4.99s    5.00s     low       0.06
walking 5.00s    12.99s   8.00s     moderate  2.14
impact  13.00s   14.98s   1.98s     severe    16.45
shaking 15.00s   18.99s   4.00s     high      6.82
```

**Example question:** "Did any impact occur?"

**Answer:**
> Yes. A severe-severity impact event was detected from 13.00s to 14.98s (duration 1.98s, peak magnitude 16.45 m/s²). This is consistent with a sudden drop, collision, or hammer blow.

**Example question:** "When was the device idle?"

**Answer:**
> The device was idle from 0.00s to 4.99s, a total of approximately 5 seconds. During this period the signal showed minimal vibration (mean 0.04 m/s²), consistent with a stationary device at rest.

---

## 13. V1 Scope

**What V1 does:**
- Ingests accelerometer CSV data or generates a synthetic 4-segment dataset
- Runs signal processing (normalisation, magnitude, rolling statistics, IQR outlier removal)
- Detects `idle`, `walking`, `impact`, `shaking`, and `unknown` events using rule-based thresholds
- Writes one plain-English summary sentence per event
- Builds a vector index with LangChain and embeds documents with `bge-small-en-v1.5`
- Orchestrates the query-answer pipeline with LangGraph
- Answers natural-language questions via a local Ollama LLM (`qwen2.5:0.5b` by default)
- Falls back to keyword matching if the LLM is unavailable
- Provides a React web UI (chat interface, event list, signal chart)
- Runs entirely on a local machine — no cloud, no API keys required

**What V1 does not do:**
- Connect to real hardware (USB, MQTT, BLE serial)
- Store historical data or event logs between sessions
- Support multi-user access or authentication
- Retrain or fine-tune any model at runtime
- Handle multi-axis threshold rules (current rules use scalar magnitude)

---

## 14. V2 Ideas

- Live hardware streaming from Bosch BMA400/BMI088 via serial (pyserial) or MQTT (paho-mqtt)
- FastAPI streaming endpoint for real-time event detection (Server-Sent Events)
- Persistent vector store (Chroma with disk storage) so history survives restarts
- Threshold auto-calibration from a 30-second baseline idle recording
- Threshold-breach alerts via email or Slack webhook
- Per-shift PDF report (event timeline + LLM summaries + severity breakdown)
- Multi-sensor comparison in one dashboard
- Docker image for edge deployment on Raspberry Pi / Jetson Nano

---

## 15. Setup

**Prerequisites:** Python 3.10+, Node.js 18+, [Ollama](https://ollama.com) installed

```bash
# 1. Clone and enter the project
git clone https://github.com/Srimonchaari/SensorSpeak.git
cd SensorSpeak

# 2. Python environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Pull the default LLM (one-time, ~350 MB)
ollama pull qwen2.5:0.5b

# 4. Start Ollama (keep this terminal open)
ollama serve

# 5. Build the React frontend (one-time)
cd frontend
npm install
npm run build
cd ..

# 6. Start the API server
python api.py
# Open http://localhost:8000
```

**For React development (hot reload):**
```bash
# Terminal 1
python api.py

# Terminal 2
cd frontend && npm run dev
# Open http://localhost:5173
```

**To switch LLM backends:**

| Backend | Command |
|---|---|
| Ollama (default, local) | `python api.py` |
| OpenAI | `export OPENAI_API_KEY=sk-... && export SENSOSPEAK_BACKEND=openai && python api.py` |
| HuggingFace API (free) | `export HF_API_KEY=hf_... && export SENSOSPEAK_BACKEND=hf_api && python api.py` |

---

## 16. Project Structure

```
SensorSpeak/
│
├── sensorspeak_core.py      ← pipeline functions (importable module)
├── llm_config.py            ← pluggable LLM backend switcher
├── api.py                   ← FastAPI server + React host
├── SensorSpeak.ipynb        ← Colab/Jupyter notebook version
├── pdf_qa.py                ← Bosch datasheet ingestion
├── finetune_prep.py         ← fine-tuning dataset generator
│
├── frontend/                ← React UI
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Navbar.tsx
│   │   │   ├── Hero.tsx
│   │   │   ├── AboutSection.tsx
│   │   │   ├── StatsGrid.tsx
│   │   │   ├── PipelinePanel.tsx
│   │   │   ├── ChatPanel.tsx
│   │   │   └── ChartView.tsx
│   │   ├── api.ts
│   │   └── types.ts
│   └── dist/                ← built output (served by api.py)
│
├── tests/
│   └── test_sensorspeak.py  ← 85 pytest unit tests
│
├── outputs/                 ← generated at runtime
│   ├── synthetic_accel_data.csv
│   ├── detected_events.csv
│   └── accelerometer_overview.png
│
├── README.md
├── QUICKSTART.md
├── CLAUDE.md
├── requirements.txt
└── .gitignore
```

---

## 17. Tunable Parameters

All parameters are named constants at the top of `sensorspeak_core.py`. Change the value and re-run.

### Event Detection Thresholds

| Parameter | Default | Tune when... |
|---|---|---|
| `IDLE_STD_MAX` | `0.15` | Sensor is noisy at rest → increase |
| `IMPACT_MEAN_MIN` | `2.5` | Missing light impacts → decrease |
| `IMPACT_STD_MIN` | `1.0` | Too many false-positive impacts → increase |
| `SHAKING_STD_MIN` | `1.8` | Walking misclassified as shaking → increase |
| `WALKING_MEAN_MIN` | `0.8` | Slow gait not detected → decrease |
| `WALKING_MEAN_MAX` | `2.5` | Fast walk classified as impact → increase |
| `MIN_EVENT_SAMPLES` | `10` | Too many short events → increase |

### LLM Settings

| Parameter | Default | Tune when... |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Want richer answers → `qwen2.5:3b` |
| `OLLAMA_TIMEOUT` | `180` | Slow machine → increase |
| `SIMILARITY_TOP_K` | `4` | More context → increase |

---

## 18. Running the Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Expected: **85 passed**.

Test coverage includes: synthetic data generation, normalisation, magnitude computation, all five event-type classification rules, event boundary detection, severity tiers, event summaries, keyword fallback query, end-to-end pipeline, and CSV error handling.

---

## 19. License

MIT — free to use, modify, and distribute with attribution.

---

*Built with Python, LangChain, LangGraph, Ollama, and a Bosch accelerometer dataset.*
