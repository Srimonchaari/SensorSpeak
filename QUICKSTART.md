# SensorSpeak — Quickstart Guide

One page. Everything you need to go from zero to a running notebook.

---

## Google Colab Setup (5 Steps)

| Step | Action |
|---|---|
| 1 | Go to [colab.research.google.com](https://colab.research.google.com/) → **File → Open notebook → Upload** → select `SensorSpeak.ipynb` |
| 2 | **Runtime → Change runtime type → T4 GPU** (free tier; speeds up embedding generation) |
| 3 | **Runtime → Run all** (`Ctrl+F9` / `Cmd+F9`) |
| 4 | Wait for Ollama install + model pull (~5 min on first run, cached on subsequent runs) |
| 5 | Scroll to **Section 10** for example query answers, **Section 11** to ask your own question |

---

## Local Setup (Python 3.10+)

```bash
# 1. Clone and enter the project
git clone https://github.com/yourname/SensorSpeak.git
cd SensorSpeak

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama  (https://ollama.com)
#    macOS/Linux:
curl -fsSL https://ollama.ai/install.sh | sh
#    Windows: download installer from ollama.com

# 5. Pull the model (one-time, ~350 MB)
ollama pull qwen2.5:0.5b

# 6. Start Ollama daemon (keep this terminal open)
ollama serve

# 7. Open the notebook in a second terminal
jupyter notebook SensorSpeak.ipynb
```

---

## Using Your Own CSV Data

Section 2 of the notebook contains a commented-out upload cell. To use real sensor data:

1. Ensure your CSV has exactly these columns: `timestamp`, `accel_x`, `accel_y`, `accel_z`
2. `timestamp` should be in seconds (float) at a consistent sample rate
3. `accel_z` should include gravity (~9.81 m/s² when the device is flat)
4. Uncomment the upload block in Section 2 and comment out the `df_raw = generate_synthetic_data()` call in Section 1

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: llama_index` | Packages not installed | Run `!pip install -r requirements.txt` in a new cell |
| `Ollama daemon not detected` | Ollama not running | Run `ollama serve` in a terminal, or uncomment `install_ollama_colab()` in Section 6 |
| `Connection refused` on port 11434 | Ollama process crashed | Restart: `pkill ollama && ollama serve` |
| Empty query answers | Model not pulled | Run `ollama pull qwen2.5:0.5b` then re-run Section 6 |
| `KeyError: '_raw_accel_x'` | Feature engineering not run | Run cells in order (Section 3 must precede Section 8) |
| `ValueError: missing columns` | CSV missing required columns | Add `timestamp`, `accel_x`, `accel_y`, `accel_z` columns |
| Very slow embedding | Running on CPU only | Switch Colab to T4 GPU runtime |
| Terse / short LLM answers | Model too small | Swap to `qwen2.5:3b` — see Model Swap Guide below |
| `No matching events found` | Keyword fallback, no overlap | Rephrase query to include event type keywords (idle, impact, shaking, walking) |
| Notebook runs but no PNG saved | `outputs/` dir not created | Re-run Section 9; the cell creates the directory automatically |

---

## Model Swap Guide

Change `OLLAMA_MODEL` at the top of **Section 6**, then re-run Sections 6, 7, and 10.

```python
# Section 6 — change this constant:
OLLAMA_MODEL = 'qwen2.5:3b'   # or 'llama3.2:1b' or 'smollm2:360m'
```

Then pull the new model before running:

```bash
ollama pull qwen2.5:3b
```

| Model | Tag | Download Size | Notes |
|---|---|---|---|
| smollm2 | `smollm2:360m` | ~220 MB | Fastest; minimal answers |
| **qwen2.5 (default)** | `qwen2.5:0.5b` | ~350 MB | Good balance of speed and quality |
| llama3.2 | `llama3.2:1b` | ~770 MB | Better reasoning; slightly slower |
| qwen2.5 | `qwen2.5:3b` | ~1.9 GB | Best quality; needs more RAM |

---

## Tuning Event Detection Thresholds

All thresholds live at the top of **Section 4**. Adjust and re-run Sections 4–11:

```python
# Raise this if your sensor is noisy at rest:
IDLE_STD_MAX      = 0.20   # default: 0.15

# Lower this if impacts are small (e.g. light taps):
IMPACT_MEAN_MIN   = 1.8    # default: 2.5

# Increase if walking-like events are misclassified as shaking:
SHAKING_STD_MIN   = 2.2    # default: 1.8
```

---

## Output Files

After a full run, `outputs/` contains:

| File | Contents |
|---|---|
| `synthetic_accel_data.csv` | Full 1900-row sensor DataFrame including features |
| `detected_events.csv` | One row per event with start/end/type/magnitude/summary |
| `accelerometer_overview.png` | 3-panel matplotlib figure (raw axes, magnitude, rolling stats) |

---

## Getting Help

- Open an issue: `https://github.com/yourname/SensorSpeak/issues`
- Check the full parameter table in `README.md` → **Tunable Parameters**
