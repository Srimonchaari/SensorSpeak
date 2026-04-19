# CLAUDE.md — SensorSpeak Project Guide

This file is read by Claude Code at the start of every session in this directory.
Always follow the rules below without needing to be reminded.

---

## Project Identity

**Name**: SensorSpeak — Bosch Accelerometer Motion Event Explainer
**Purpose**: A fully local Python pipeline that ingests raw accelerometer CSV data,
detects motion events with rule-based logic, generates plain-English summaries,
and answers natural-language queries via LlamaIndex + Ollama.

---

## Non-Negotiable Constraints

These rules must never be broken, regardless of what is asked:

1. **No cloud services** — no OpenAI, Anthropic API, Google AI, Azure, AWS Bedrock, or any hosted LLM API
2. **No API keys** — the project must run with zero credentials
3. **No deep learning training** — no fine-tuning, no LoRA, no custom model training
4. **No seaborn** — matplotlib only for all visualisation
5. **No new ML model training** — rule-based detection only; thresholds are tuned, not learned
6. **Ollama is the only LLM backend** — model: `qwen2.5:0.5b` (default); alternatives listed in README Model Options table
7. **HuggingFace bge-small-en-v1.5 is the only embedding model** — loaded locally via `sentence-transformers`
8. **Python 3.10 only** — no f-string walrus operators or 3.11+ syntax

---

## Stack (frozen — do not add new dependencies without asking)

| Layer | Tool | Version |
|---|---|---|
| Language | Python | 3.10 |
| Data | pandas | >=2.0.0 |
| Numerics | numpy | >=1.24.0 |
| Visualisation | matplotlib | >=3.7.0 |
| LLM framework | llama-index-core | >=0.10.0 |
| LLM backend | llama-index-llms-ollama | >=0.1.0 |
| Embeddings | llama-index-embeddings-huggingface | >=0.1.0 |
| Embedding model | sentence-transformers | >=2.2.0 |
| LLM server | Ollama | latest |
| LLM model | qwen2.5:0.5b | — |
| Embedding model | BAAI/bge-small-en-v1.5 | — |

---

## CSV Schema (immutable)

```
timestamp   float   seconds since recording start
accel_x     float   m/s²  (lateral)
accel_y     float   m/s²  (forward)
accel_z     float   m/s²  (vertical; includes ~9.81 gravity offset)
```

Synthetic dataset also includes `_segment_label` (idle / walking_like / impact / shaking).
All `_raw_*` columns are preserved post-normalisation.

---

## Pipeline Order (must be maintained)

```
Section 1  generate_synthetic_data()
Section 2  optional CSV upload (commented out)
Section 3  normalize_and_engineer_features()
Section 4  detect_events()  →  List[MotionEvent]
Section 5  summarize_event()  →  List[str]
Section 6  build_index()  →  VectorStoreIndex | None
Section 7  query_events()
Section 8  visualize()
Section 9  save_outputs()
Section 10 pipeline summary
Section 11 MY_QUESTION interactive cell
```

Later sections depend on earlier ones. Never re-order them.

---

## Coding Rules

- **All thresholds as named constants** at the top of their cell — never magic numbers inline
- **Modular functions only** — no logic outside function bodies except the final call per cell
- **Keyword fallback is always present** — `query_events()` must work without Ollama running
- **Safe column access** — always validate required columns with a clear `ValueError` before use
- **Single-quotes preferred** in Python string literals to keep notebook JSON clean
- **No comments on obvious lines** — only comment non-obvious behaviour or hidden constraints
- **No trailing summaries** in generated code cells — let the print statements speak

---

## Named Constants Reference (default values)

### Data Generation
```python
SAMPLE_RATE_HZ       = 100
GRAVITY_Z            = 9.81
SEG_IDLE_DURATION    = 5
SEG_WALK_DURATION    = 8
SEG_IMPACT_DURATION  = 2
SEG_SHAKE_DURATION   = 4
IDLE_NOISE           = 0.05
WALK_AMPLITUDE       = 1.2
WALK_FREQ_HZ         = 2.0
WALK_NOISE           = 0.2
IMPACT_SPIKE         = 12.0
IMPACT_NOISE         = 0.5
SHAKE_AMPLITUDE      = 4.0
SHAKE_FREQ_HZ        = 8.0
SHAKE_NOISE          = 0.8
RANDOM_SEED          = 42
```

### Signal Processing
```python
ROLLING_WINDOW       = 50
ZSCORE_EPS           = 1e-8
```

### Event Detection
```python
IDLE_STD_MAX         = 0.15
IMPACT_MEAN_MIN      = 2.5
IMPACT_STD_MIN       = 1.0
SHAKING_STD_MIN      = 1.8
WALKING_MEAN_MIN     = 0.8
WALKING_MEAN_MAX     = 2.5
WALKING_STD_MIN      = 0.10
WALKING_STD_MAX      = 1.8
MIN_EVENT_SAMPLES    = 10
```

### LLM / Index
```python
OLLAMA_MODEL         = 'qwen2.5:0.5b'
OLLAMA_TIMEOUT       = 180
EMBED_MODEL_NAME     = 'BAAI/bge-small-en-v1.5'
SIMILARITY_TOP_K     = 4
RESPONSE_MODE        = 'compact'
```

---

## Event Types (do not add new types without updating all downstream cells)

| Type | Rule |
|---|---|
| `idle` | rolling_std < IDLE_STD_MAX |
| `impact` | rolling_mean > IMPACT_MEAN_MIN AND rolling_std > IMPACT_STD_MIN |
| `shaking` | rolling_std > SHAKING_STD_MIN |
| `walking` | WALKING_MEAN_MIN ≤ rolling_mean ≤ WALKING_MEAN_MAX AND WALKING_STD_MIN ≤ rolling_std ≤ WALKING_STD_MAX |
| `unknown` | none of the above |

---

## Output Files (paths are fixed)

```
outputs/synthetic_accel_data.csv
outputs/detected_events.csv
outputs/accelerometer_overview.png
```

---

## Roadmap Context

When suggesting or implementing improvements, keep these phases in mind:

- **Phase 1 (current)**: offline Colab notebook — DONE
- **Phase 2 (next)**: live hardware streaming + FastAPI `/query` endpoint
- **Phase 3 (future)**: Streamlit dashboard + threshold-breach alerts + PDF reports

Do not implement Phase 2 or 3 features unless explicitly requested.

---

## Files in This Repository

| File | Purpose |
|---|---|
| `SensorSpeak.ipynb` | Main deliverable — full Colab notebook |
| `README.md` | Project documentation for GitHub and recruiters |
| `requirements.txt` | Pinned Python dependencies |
| `QUICKSTART.md` | One-page setup + troubleshooting guide |
| `CLAUDE.md` | This file — Claude Code session context |
| `agent.md` | Agent behaviour rules for autonomous tasks |
| `Bosch_Accelerometer_Motion_Event_Explainer.ipynb` | Earlier prototype (reference only) |
| `Bosch_Accelerometer_Ollama_QueryEngine.ipynb` | Earlier prototype (reference only) |
