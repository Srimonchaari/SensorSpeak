# SensorSpeak — Bosch Accelerometer Motion Event Explainer

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20qwen2.5%3A0.5b-purple)
![Framework](https://img.shields.io/badge/Framework-LlamaIndex-orange)
![No Cloud](https://img.shields.io/badge/Cloud-None-red)
![No API Keys](https://img.shields.io/badge/API%20Keys-None-red)

> Turn raw accelerometer CSV data into plain-English motion event reports — fully local, no cloud, no API keys, no ML training required.

---

## Live Demo

**[srimonchaari.github.io/SensorSpeak](https://srimonchaari.github.io/SensorSpeak/)**

The demo site gives a visual walkthrough of the pipeline, event types, and example Q&A output. No installation needed to explore the project.

---

## Table of Contents

1. [Overview](#1-overview)
2. [How It Works](#2-how-it-works)
3. [Pipeline](#3-pipeline)
4. [Event Types](#4-event-types)
5. [Example Input & Output](#5-example-input--output)
6. [Tech Stack](#6-tech-stack)
7. [Setup](#7-setup)
8. [Project Structure](#8-project-structure)
9. [Tunable Parameters](#9-tunable-parameters)
10. [Running the Tests](#10-running-the-tests)
11. [Roadmap](#11-roadmap)
12. [License](#12-license)

---

## 1. Overview

**SensorSpeak** ingests a CSV file from any Bosch MEMS accelerometer (BMA400, BMI088, etc.) — or generates a synthetic dataset automatically — and runs a three-stage local pipeline:

| Stage | What happens |
|---|---|
| **Detect** | Rule-based signal processing classifies every sample into `idle`, `walking`, `impact`, `shaking`, or `unknown` |
| **Explain** | Each detected event becomes one plain-English sentence: time range, duration, severity, peak value, and physical interpretation |
| **Answer** | A local LLM (Ollama + LlamaIndex) answers natural-language questions about those events; keyword fallback works without Ollama |

You can ask questions like:
- *"Did any impact occur?"*
- *"When was the device idle?"*
- *"Was there any shaking, and how severe was it?"*
- *"What was the peak acceleration?"*

Everything runs on your machine. No data leaves your environment.

---

## 2. How It Works

### Accelerometer basics

An accelerometer measures acceleration in three orthogonal axes simultaneously. Bosch MEMS accelerometers output a continuous stream at a fixed sample rate (typically 100 Hz — one row every 10 ms).

| Column | Axis | Typical at rest |
|---|---|---|
| `accel_x` | Lateral (left/right) | ≈ 0.00 m/s² |
| `accel_y` | Forward (front/back) | ≈ 0.00 m/s² |
| `accel_z` | Vertical (up/down) + gravity | ≈ 9.81 m/s² |

`accel_z` always reads ~9.81 m/s² when the device is flat and still — this is the gravitational offset, not motion noise.

### Signal processing

The three axes are Z-score normalised and combined into a single **magnitude** (`√(x²+y²+z²)`), which captures total motion energy independent of direction. A rolling window (50 samples) then computes the mean and standard deviation of that magnitude. These two values are the inputs to the rule engine.

### Rule-based detection

Fixed thresholds on `rolling_mean` and `rolling_std` classify each sample. No training data, no labelled examples, no model weights. Consecutive samples with the same label (≥10 samples) become a `MotionEvent` object.

### LlamaIndex query layer

Event summaries are loaded as `Document` objects, embedded with `BAAI/bge-small-en-v1.5` (local via sentence-transformers), and stored in a `VectorStoreIndex`. When a question arrives, LlamaIndex retrieves the top-k most relevant summaries and passes them to Ollama for answer synthesis. If Ollama is not running, the system falls back to keyword matching over the raw summaries.

---

## 3. Pipeline

```
Raw CSV  (timestamp, accel_x, accel_y, accel_z)
         │
         ▼
┌─────────────────────────────────────┐
│  Signal Processing                  │
│  · Z-score normalise x, y, z        │
│  · magnitude = √(x² + y² + z²)     │
│  · rolling_mean  (window = 50)      │
│  · rolling_std   (window = 50)      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  IQR Outlier Removal                │
│  · Winsorise magnitude at fences    │
│  · Recompute rolling stats          │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Rule-Based Event Detector          │
│                                     │
│  idle    → std < 0.15               │
│  impact  → mean > 2.5 AND std > 1.0 │
│  shaking → std > 1.8                │
│  walking → mean ∈ [0.8, 2.5]        │
│            AND std ∈ [0.10, 1.8]    │
│  unknown → none of the above        │
│                                     │
│  ≥ 10 consecutive samples → Event   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Event Summariser                   │
│  One plain-English sentence/event   │
│  (time range, duration, severity,   │
│   peak, physical interpretation)    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  LlamaIndex Vector Store            │
│  · Embed summaries (bge-small-en)   │
│  · Store in VectorStoreIndex        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Query Engine                       │
│  · Retrieve top-k relevant docs     │
│  · Ollama LLM synthesises answer    │
│  · Keyword fallback if no LLM       │
└──────────────────┬──────────────────┘
                   │
                   ▼
          Plain-English Answer
```

---

## 4. Event Types

| Event | Detection Rule | Physical Meaning |
|---|---|---|
| `idle` | `rolling_std < 0.15` | Stationary — machine off, device at rest |
| `walking` | `mean ∈ [0.8, 2.5]` and `std ∈ [0.10, 1.8]` | Rhythmic periodic motion — walking, rotating shaft, belt conveyor |
| `impact` | `mean > 2.5` and `std > 1.0` | Sudden high-energy transient — drop, collision, tool strike |
| `shaking` | `std > 1.8` | Rapid sustained oscillation — loose component, bearing fault |
| `unknown` | none of the above | State transition; thresholds may need tuning |

### Severity tiers (by peak magnitude)

| Severity | Peak magnitude |
|---|---|
| `low` | < 1.5 m/s² |
| `moderate` | 1.5 – 3.5 m/s² |
| `high` | 3.5 – 7.0 m/s² |
| `severe` | > 7.0 m/s² |

---

## 5. Example Input & Output

**Input CSV** (100 Hz — one row per 10 ms):

```csv
timestamp,accel_x,accel_y,accel_z
0.00, 0.02,-0.01, 9.82
0.01, 0.03, 0.00, 9.80
...
13.00, 8.42, 3.17,15.63
13.01, 6.21, 2.88,14.22
```

**Detected events:**

```
Event    Start    End      Duration  Severity  Peak
idle     0.00s    4.99s    5.00s     low       0.06
walking  5.00s    12.99s   8.00s     moderate  2.14
impact   13.00s   14.98s   1.98s     severe    16.45
shaking  15.00s   18.99s   4.00s     high      6.82
```

**Q: "Did any impact occur?"**
> Yes. A severe-severity impact event was detected from 13.00 s to 14.98 s (duration 1.98 s, peak magnitude 16.45 m/s²). This is consistent with a sudden drop, collision, or hammer blow.

**Q: "When was the device idle?"**
> The device was idle from 0.00 s to 4.99 s — approximately 5 seconds. Signal showed minimal vibration (mean 0.04 m/s²), consistent with a stationary device at rest.

---

## 6. Tech Stack

| Layer | Tool | Version |
|---|---|---|
| Language | Python | 3.10 |
| Data | pandas | ≥ 2.0 |
| Numerics | numpy | ≥ 1.24 |
| Visualisation | matplotlib | ≥ 3.7 |
| LLM framework | LlamaIndex (`llama-index-core`) | ≥ 0.10 |
| LLM backend | `llama-index-llms-ollama` | ≥ 0.1 |
| Embeddings | `llama-index-embeddings-huggingface` | ≥ 0.1 |
| Embedding model | `BAAI/bge-small-en-v1.5` (sentence-transformers) | local |
| LLM server | Ollama | latest |
| Default LLM | `qwen2.5:0.5b` | local |

### LLM backend options

| Backend | Requirement |
|---|---|
| Ollama (default, fully local) | `ollama pull qwen2.5:0.5b` |
| OpenAI | `OPENAI_API_KEY` env var |
| HuggingFace Inference API | `HF_API_KEY` env var (free tier available) |
| HuggingFace local | Any causal LM via `transformers` |

The LLM only synthesises answers. All event classification happens in the rule engine before the LLM is ever called. The system answers correctly even without Ollama running.

---

## 7. Setup

**Prerequisites:** Python 3.10, Node.js 18+, [Ollama](https://ollama.com)

```bash
# 1. Clone
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
cd frontend && npm install && npm run build && cd ..

# 6. Start the API server
python api.py
# → http://localhost:8000
```

**Frontend hot-reload (development):**

```bash
# Terminal 1
python api.py

# Terminal 2
cd frontend && npm run dev
# → http://localhost:5173
```

**Notebook-only (no server needed):**

Open `SensorSpeak.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

**Switch LLM backend:**

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
export SENSORSPEAK_BACKEND=openai
python api.py

# HuggingFace Inference API
export HF_API_KEY=hf_...
export SENSORSPEAK_BACKEND=hf_api
python api.py
```

---

## 8. Project Structure

```
SensorSpeak/
├── sensorspeak_core.py      ← all pipeline functions (importable)
├── llm_config.py            ← pluggable LLM backend switcher
├── api.py                   ← FastAPI server + React host
├── SensorSpeak.ipynb        ← full Colab/Jupyter notebook
├── pdf_qa.py                ← Bosch datasheet Q&A ingestion
├── finetune_prep.py         ← instruction dataset generator
│
├── frontend/                ← React UI (Vite + TypeScript + Tailwind)
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Navbar.tsx
│   │   │   ├── Hero.tsx
│   │   │   ├── PipelinePanel.tsx
│   │   │   ├── ChatPanel.tsx
│   │   │   └── ChartView.tsx
│   │   ├── api.ts
│   │   └── types.ts
│   └── dist/                ← built output (served by api.py)
│
├── docs/                    ← GitHub Pages demo site
│   └── index.html
│
├── tests/
│   └── test_sensorspeak.py  ← 85 pytest unit tests
│
├── outputs/                 ← generated at runtime
│   ├── synthetic_accel_data.csv
│   ├── detected_events.csv
│   └── accelerometer_overview.png
│
├── requirements.txt
├── QUICKSTART.md
├── CLAUDE.md
└── .gitignore
```

---

## 9. Tunable Parameters

All parameters are named constants at the top of `sensorspeak_core.py`. Change the value and re-run — no other files need editing.

### Event detection thresholds

| Parameter | Default | When to change |
|---|---|---|
| `IDLE_STD_MAX` | `0.15` | Sensor is noisy at rest → increase |
| `IMPACT_MEAN_MIN` | `2.5` | Missing light impacts → decrease |
| `IMPACT_STD_MIN` | `1.0` | Too many false-positive impacts → increase |
| `SHAKING_STD_MIN` | `1.8` | Walking misclassified as shaking → increase |
| `WALKING_MEAN_MIN` | `0.8` | Slow gait not detected → decrease |
| `WALKING_MEAN_MAX` | `2.5` | Fast walk classified as impact → increase |
| `MIN_EVENT_SAMPLES` | `10` | Too many short spurious events → increase |

### LLM settings

| Parameter | Default | When to change |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Want richer answers → `qwen2.5:3b` |
| `OLLAMA_TIMEOUT` | `180` | Slow machine → increase |
| `SIMILARITY_TOP_K` | `4` | More context in answers → increase |

---

## 10. Running the Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Expected: **85 passed**.

Coverage includes: synthetic data generation, normalisation, magnitude computation, all five event-type rules, event boundary detection, all four severity tiers, event summarisation, keyword fallback query, end-to-end pipeline, and CSV error handling.

---

## 11. Roadmap

### V1 — Complete

- Synthetic 4-segment dataset (idle → walking → impact → shaking)
- Signal processing: normalisation, magnitude, rolling statistics, IQR outlier removal
- Rule-based event detection (5 event types, 4 severity tiers)
- Plain-English event summaries
- LlamaIndex vector store + Ollama query engine
- Keyword fallback when Ollama is unavailable
- React web UI (drag-drop upload, chat, event list, signal chart)
- FastAPI REST server (`/api/run`, `/api/chat`, `/api/health`)
- 85 pytest unit tests
- GitHub Pages demo site

### V2 — Planned

- Live hardware streaming from Bosch BMA400/BMI088 via serial or MQTT
- Persistent vector store (Chroma) — history survives restarts
- Threshold auto-calibration from a 30-second idle baseline recording
- Threshold-breach alerts via email or Slack webhook
- Per-shift PDF report (event timeline + LLM summaries + severity chart)
- Docker image for edge deployment (Raspberry Pi / Jetson Nano)
- Multi-sensor comparison in one dashboard

---

## 12. License

MIT — free to use, modify, and distribute with attribution.

---

*Built with Python · LlamaIndex · Ollama · sentence-transformers · React · FastAPI*
