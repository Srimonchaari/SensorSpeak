"""
sensorspeak_core.py

All pipeline functions extracted from SensorSpeak.ipynb.
Import this module from unit tests, the Gradio UI, and the PDF QA tool.
"""

import numpy as np
import pandas as pd
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

# ── DATA GENERATION CONSTANTS ─────────────────────────────────────────────────
SAMPLE_RATE_HZ        = 100
GRAVITY_Z             = 9.81
SEG_IDLE_DURATION     = 5
SEG_WALK_DURATION     = 8
SEG_IMPACT_DURATION   = 2
SEG_SHAKE_DURATION    = 4
IDLE_NOISE            = 0.05
WALK_AMPLITUDE        = 1.2
WALK_FREQ_HZ          = 2.0
WALK_NOISE            = 0.2
IMPACT_SPIKE          = 12.0
IMPACT_NOISE          = 0.5
SHAKE_AMPLITUDE       = 4.0
SHAKE_FREQ_HZ         = 8.0
SHAKE_NOISE           = 0.8
RANDOM_SEED           = 42

# ── SIGNAL PROCESSING CONSTANTS ───────────────────────────────────────────────
ROLLING_WINDOW        = 50
ZSCORE_EPS            = 1e-8

# ── IQR OUTLIER REMOVAL CONSTANTS ─────────────────────────────────────────────
IQR_MULTIPLIER        = 1.5    # Tukey fence (1.5 = standard; 3.0 = extreme-only)
IQR_APPLY_LOWER       = True   # clip sensor dropouts / dead-zone glitches
IQR_APPLY_UPPER       = False  # keep high-energy events (impacts, shaking) intact

# ── EVENT DETECTION THRESHOLDS ────────────────────────────────────────────────
IDLE_STD_MAX          = 0.15
IMPACT_MEAN_MIN       = 2.5
IMPACT_STD_MIN        = 1.0
SHAKING_STD_MIN       = 1.8
WALKING_MEAN_MIN      = 0.8
WALKING_MEAN_MAX      = 2.5
WALKING_STD_MIN       = 0.10
WALKING_STD_MAX       = 1.8
MIN_EVENT_SAMPLES     = 10

# ── LLM / INDEX SETTINGS ──────────────────────────────────────────────────────
OLLAMA_MODEL          = 'qwen2.5:0.5b'
OLLAMA_TIMEOUT        = 180
EMBED_MODEL_NAME      = 'BAAI/bge-small-en-v1.5'
SIMILARITY_TOP_K      = 4
RESPONSE_MODE         = 'compact'

# ── OUTPUT PATHS ──────────────────────────────────────────────────────────────
OUTPUT_DIR            = 'outputs'
CSV_RAW_FILE          = os.path.join(OUTPUT_DIR, 'synthetic_accel_data.csv')
CSV_EVENTS_FILE       = os.path.join(OUTPUT_DIR, 'detected_events.csv')
VIZ_OUTPUT_FILE       = os.path.join(OUTPUT_DIR, 'accelerometer_overview.png')

EVENT_COLORS = {
    'idle':    '#90EE90',
    'walking': '#87CEEB',
    'impact':  '#FF6B6B',
    'shaking': '#FFD700',
    'unknown': '#D3D3D3',
}

_SEED_MAP = {
    'idle':    'minimal vibration; device appears to be at rest',
    'walking': 'rhythmic oscillation consistent with footsteps or a rotating shaft',
    'impact':  'sharp high-amplitude spike; possible drop, collision, or hammer blow',
    'shaking': 'rapid high-frequency oscillation indicating intentional agitation or loose component',
    'unknown': 'unclassified motion pattern; threshold tuning may be required',
}


@dataclass
class MotionEvent:
    start:    float
    end:      float
    type:     str
    max_mag:  float
    mean_mag: float
    seed:     str


# ── Section 1: Data Generation ────────────────────────────────────────────────

def generate_synthetic_data() -> pd.DataFrame:
    """Generate a 4-segment accelerometer dataset at SAMPLE_RATE_HZ."""
    rng = np.random.default_rng(RANDOM_SEED)
    segments = []

    def _make_time(duration: float) -> np.ndarray:
        n = int(duration * SAMPLE_RATE_HZ)
        return np.linspace(0, duration, n, endpoint=False)

    t = _make_time(SEG_IDLE_DURATION)
    ax = rng.normal(0, IDLE_NOISE, len(t))
    ay = rng.normal(0, IDLE_NOISE, len(t))
    az = rng.normal(GRAVITY_Z, IDLE_NOISE, len(t))
    segments.append(pd.DataFrame({'accel_x': ax, 'accel_y': ay, 'accel_z': az,
                                   '_segment_label': 'idle'}))

    t = _make_time(SEG_WALK_DURATION)
    ax = WALK_AMPLITUDE * np.sin(2 * np.pi * WALK_FREQ_HZ * t) + rng.normal(0, WALK_NOISE, len(t))
    ay = (WALK_AMPLITUDE * 0.5) * np.cos(2 * np.pi * WALK_FREQ_HZ * t) + rng.normal(0, WALK_NOISE, len(t))
    az = rng.normal(GRAVITY_Z, WALK_NOISE, len(t))
    segments.append(pd.DataFrame({'accel_x': ax, 'accel_y': ay, 'accel_z': az,
                                   '_segment_label': 'walking_like'}))

    t = _make_time(SEG_IMPACT_DURATION)
    ax = rng.normal(0, IMPACT_NOISE, len(t))
    ay = rng.normal(0, IMPACT_NOISE, len(t))
    az = rng.normal(GRAVITY_Z, IMPACT_NOISE, len(t))
    mid = len(t) // 2
    ax[mid - 2 : mid + 3] += IMPACT_SPIKE
    az[mid - 2 : mid + 3] += IMPACT_SPIKE
    segments.append(pd.DataFrame({'accel_x': ax, 'accel_y': ay, 'accel_z': az,
                                   '_segment_label': 'impact'}))

    t = _make_time(SEG_SHAKE_DURATION)
    ax = SHAKE_AMPLITUDE * np.sin(2 * np.pi * SHAKE_FREQ_HZ * t) + rng.normal(0, SHAKE_NOISE, len(t))
    ay = SHAKE_AMPLITUDE * np.cos(2 * np.pi * SHAKE_FREQ_HZ * t) + rng.normal(0, SHAKE_NOISE, len(t))
    az = rng.normal(GRAVITY_Z, SHAKE_NOISE, len(t))
    segments.append(pd.DataFrame({'accel_x': ax, 'accel_y': ay, 'accel_z': az,
                                   '_segment_label': 'shaking'}))

    df = pd.concat(segments, ignore_index=True)
    df.insert(0, 'timestamp', np.arange(len(df)) / SAMPLE_RATE_HZ)
    return df


# ── Section 3: Feature Engineering ───────────────────────────────────────────

def normalize_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize axes; add magnitude and rolling statistics."""
    required = {'accel_x', 'accel_y', 'accel_z'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Input DataFrame is missing required columns: {missing}')

    df = df.copy()
    for axis in ('accel_x', 'accel_y', 'accel_z'):
        df[f'_raw_{axis}'] = df[axis]
        mean_val = df[axis].mean()
        std_val  = df[axis].std()
        df[axis] = (df[axis] - mean_val) / (std_val + ZSCORE_EPS)

    df['accel_magnitude'] = np.sqrt(
        df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2
    )
    df['rolling_mean'] = (
        df['accel_magnitude']
        .rolling(window=ROLLING_WINDOW, min_periods=1, center=True)
        .mean()
    )
    df['rolling_std'] = (
        df['accel_magnitude']
        .rolling(window=ROLLING_WINDOW, min_periods=1, center=True)
        .std()
        .fillna(0)
    )
    return df


# ── Section 3b: IQR Outlier Removal ──────────────────────────────────────────

def remove_outliers_iqr(df: pd.DataFrame) -> tuple:
    """
    Winsorise accel_magnitude using the Tukey IQR fence.

    - Lower fence  (Q1 - IQR_MULTIPLIER * IQR): clips sensor dropouts and
      dead-zone glitches that read near-zero or negative.
    - Upper fence  is NOT applied by default — impacts and shaking produce
      legitimately high magnitudes that must reach detect_events() intact.

    Rolling mean and std are recomputed on the cleaned magnitude so that
    the event detector sees clean statistics.

    Returns
    -------
    cleaned_df : pd.DataFrame  (same length, magnitudes winsorised)
    report     : dict          (Q1, Q3, IQR, fences, n_clipped, pct_clipped)
    """
    required = {'accel_magnitude', 'rolling_mean', 'rolling_std'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f'Missing columns: {missing}. Run normalize_and_engineer_features first.'
        )

    df = df.copy()
    mag = df['accel_magnitude']

    q1  = float(mag.quantile(0.25))
    q3  = float(mag.quantile(0.75))
    iqr = q3 - q1

    lower_fence = q1 - IQR_MULTIPLIER * iqr
    upper_fence = q3 + IQR_MULTIPLIER * iqr

    # Flag which samples fall outside the fences before clipping
    below = mag < lower_fence
    above = mag > upper_fence
    df['_iqr_outlier'] = below | (above & IQR_APPLY_UPPER)

    # Winsorise: clip only the chosen fences
    clip_lo = max(lower_fence, 0.0) if IQR_APPLY_LOWER else None
    clip_hi = upper_fence           if IQR_APPLY_UPPER else None
    df['accel_magnitude'] = mag.clip(lower=clip_lo, upper=clip_hi)

    # Recompute rolling stats on cleaned magnitude
    df['rolling_mean'] = (
        df['accel_magnitude']
        .rolling(window=ROLLING_WINDOW, min_periods=1, center=True)
        .mean()
    )
    df['rolling_std'] = (
        df['accel_magnitude']
        .rolling(window=ROLLING_WINDOW, min_periods=1, center=True)
        .std()
        .fillna(0)
    )

    n_clipped = int(df['_iqr_outlier'].sum())
    report = {
        'q1':           round(q1, 4),
        'q3':           round(q3, 4),
        'iqr':          round(iqr, 4),
        'lower_fence':  round(lower_fence, 4),
        'upper_fence':  round(upper_fence, 4),
        'n_clipped':    n_clipped,
        'pct_clipped':  round(n_clipped / len(df) * 100, 2),
        'rows_total':   len(df),
    }

    return df, report


# ── Section 4: Event Detection ────────────────────────────────────────────────

def _classify_sample(rmean: float, rstd: float) -> str:
    """Apply threshold rules to one (rolling_mean, rolling_std) pair."""
    if rstd < IDLE_STD_MAX:
        return 'idle'
    if rmean > IMPACT_MEAN_MIN and rstd > IMPACT_STD_MIN:
        return 'impact'
    if rstd > SHAKING_STD_MIN:
        return 'shaking'
    if WALKING_MEAN_MIN <= rmean <= WALKING_MEAN_MAX and WALKING_STD_MIN <= rstd <= WALKING_STD_MAX:
        return 'walking'
    return 'unknown'


def detect_events(df: pd.DataFrame) -> List[MotionEvent]:
    """Segment the DataFrame into MotionEvent objects."""
    for col in ('timestamp', 'rolling_mean', 'rolling_std', 'accel_magnitude'):
        if col not in df.columns:
            raise ValueError(f'Required column "{col}" is missing from DataFrame.')

    labels = [
        _classify_sample(rm, rs)
        for rm, rs in zip(df['rolling_mean'], df['rolling_std'])
    ]

    events: List[MotionEvent] = []
    i, n = 0, len(labels)
    while i < n:
        current_label = labels[i]
        j = i
        while j < n and labels[j] == current_label:
            j += 1
        if (j - i) >= MIN_EVENT_SAMPLES:
            window = df.iloc[i:j]
            events.append(MotionEvent(
                start    = float(window['timestamp'].iloc[0]),
                end      = float(window['timestamp'].iloc[-1]),
                type     = current_label,
                max_mag  = float(window['accel_magnitude'].max()),
                mean_mag = float(window['accel_magnitude'].mean()),
                seed     = _SEED_MAP[current_label],
            ))
        i = j
    return events


# ── Section 5: Summarizer ─────────────────────────────────────────────────────

def _severity_label(max_mag: float) -> str:
    """Map peak magnitude to a severity tier."""
    if max_mag < 1.5:
        return 'low'
    if max_mag < 3.5:
        return 'moderate'
    if max_mag < 7.0:
        return 'high'
    return 'severe'


def summarize_event(event: MotionEvent) -> str:
    """Return a single-sentence plain-English description of a MotionEvent."""
    duration = event.end - event.start
    sev = _severity_label(event.max_mag)
    return (
        f'From {event.start:.2f}s to {event.end:.2f}s ({duration:.2f}s), '
        f'a {sev}-severity "{event.type}" event was detected '
        f'(peak magnitude {event.max_mag:.4f}, mean {event.mean_mag:.4f} m/s\u00b2) '
        f'\u2014 {event.seed}.'
    )


# ── Section 6/7: Index + Query ────────────────────────────────────────────────

def _is_ollama_running() -> bool:
    """Return True if the local Ollama daemon responds on port 11434."""
    try:
        urllib.request.urlopen('http://localhost:11434', timeout=3)
        return True
    except Exception:
        return False


def build_index(events: List[MotionEvent], summaries: List[str],
                llm=None, embed_model=None):
    """
    Build a VectorStoreIndex from event summaries.

    Pass custom llm / embed_model to override the Ollama defaults (e.g. OpenAI).
    Returns None and falls back to keyword search if dependencies are unavailable.
    """
    try:
        from llama_index.core import VectorStoreIndex, Document, Settings
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        print(f'LlamaIndex import failed ({e}) — using keyword fallback.')
        return None

    # Use provided backends or default to Ollama + HuggingFace
    if llm is None:
        if not _is_ollama_running():
            print('Ollama not running — using keyword fallback.')
            return None
        llm = Ollama(model=OLLAMA_MODEL, request_timeout=OLLAMA_TIMEOUT)
    if embed_model is None:
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    try:
        Settings.llm = llm
        Settings.embed_model = embed_model
        docs = [
            __import__('llama_index.core', fromlist=['Document']).Document(
                text=summary,
                metadata={'event_type': ev.type, 'start': ev.start,
                          'end': ev.end, 'max_mag': round(ev.max_mag, 4)}
            )
            for ev, summary in zip(events, summaries)
        ]
        index = VectorStoreIndex.from_documents(docs)
        print(f'VectorStoreIndex ready: {len(docs)} documents.')
        return index
    except Exception as exc:
        print(f'Index build failed ({exc}) — using keyword fallback.')
        return None


def _keyword_fallback(question: str, summaries: List[str]) -> str:
    """Intent-aware keyword fallback when Ollama is not running."""
    import re as _re
    q = question.lower()
    note = '> *Ollama not running — using keyword fallback. Start Ollama for full NL answers.*\n\n'

    if not summaries:
        return note + 'No events available. Run the pipeline first.'

    def _bullets(items):
        return '\n'.join(f'• {s}' for s in items)

    def _max_mag(s):
        m = _re.search(r'peak magnitude ([\d.]+)', s)
        return float(m.group(1)) if m else 0.0

    # ── Intent: dangerous / high-severity ────────────────────────────────────
    if any(w in q for w in ('danger', 'severe', 'critical', 'high', 'worst', 'alert', 'bad')):
        matched = [s for s in summaries if any(k in s.lower() for k in ('impact', 'shaking', 'high', 'severe', 'critical'))]
        if not matched:
            return note + 'No high-severity events (impact or shaking) were detected in this recording.'
        return note + f'**{len(matched)} high-severity event(s) detected:**\n\n' + _bullets(matched)

    # ── Intent: specific event type ───────────────────────────────────────────
    type_map = {'idle': 'idle', 'rest': 'idle', 'still': 'idle',
                'walk': 'walking', 'moving': 'walking', 'motion': 'walking',
                'impact': 'impact', 'hit': 'impact', 'collision': 'impact', 'drop': 'impact',
                'shak': 'shaking', 'vibrat': 'shaking', 'tremor': 'shaking'}
    for kw, ev_type in type_map.items():
        if kw in q:
            matched = [s for s in summaries if ev_type in s.lower()]
            if not matched:
                return note + f'No **{ev_type}** events were detected in this recording.'
            return note + f'**{len(matched)} {ev_type} event(s):**\n\n' + _bullets(matched)

    # ── Intent: peak / maximum ────────────────────────────────────────────────
    if any(w in q for w in ('peak', 'max', 'highest', 'strongest', 'largest', 'biggest')):
        top = max(summaries, key=_max_mag, default=None)
        if not top:
            return note + 'No magnitude data found in events.'
        return note + f'**Highest-magnitude event:**\n\n• {top}'

    # ── Intent: duration / time ───────────────────────────────────────────────
    if any(w in q for w in ('long', 'duration', 'how long', 'last', 'period', 'time', 'seconds')):
        return note + f'**All {len(summaries)} events with timing:**\n\n' + _bullets(summaries)

    # ── Intent: summary / list all ────────────────────────────────────────────
    if any(w in q for w in ('all', 'every', 'summarise', 'summarize', 'overview', 'list', 'describe', 'order', 'happened')):
        return note + f'**Full event log ({len(summaries)} events):**\n\n' + _bullets(summaries)

    # ── General: keyword scoring (skip stop-words) ────────────────────────────
    stop = {'the','a','an','is','are','was','were','any','there','what','how',
            'when','did','do','i','me','my','in','of','to','for','and','or'}
    q_terms = set(q.split()) - stop
    if q_terms:
        scored = [(sum(t in s.lower() for t in q_terms), s) for s in summaries]
        scored = [(sc, s) for sc, s in scored if sc > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            return note + _bullets(s for _, s in scored[:SIMILARITY_TOP_K])

    return note + f'No close matches found. Here is the full event log:\n\n' + _bullets(summaries)


def query_events(question: str, index, summaries: List[str],
                 verbose: bool = False) -> str:
    """Query motion events with a natural-language question."""
    if not question.strip():
        return 'Please provide a non-empty question.'
    if index is not None:
        try:
            qe = index.as_query_engine(
                similarity_top_k=SIMILARITY_TOP_K,
                response_mode=RESPONSE_MODE,
            )
            response = qe.query(question)
            if verbose:
                print(f'[LlamaIndex] {len(response.source_nodes)} source nodes retrieved.')
            return str(response)
        except Exception as exc:
            if verbose:
                print(f'LLM query error: {exc}')
    return _keyword_fallback(question, summaries)


# ── Full pipeline runner ───────────────────────────────────────────────────────

def run_pipeline(csv_path: Optional[str] = None, llm=None, embed_model=None):
    """
    Run the complete SensorSpeak pipeline end-to-end.

    Args:
        csv_path:    Path to a real CSV file. Uses synthetic data if None.
        llm:         Optional LlamaIndex LLM override (e.g. OpenAI instance).
        embed_model: Optional embedding model override.

    Returns:
        dict with keys: df, events, summaries, index
    """
    if csv_path:
        df_raw = pd.read_csv(csv_path)
        required = {'timestamp', 'accel_x', 'accel_y', 'accel_z'}
        missing = required - set(df_raw.columns)
        if missing:
            raise ValueError(f'CSV missing required columns: {missing}')
    else:
        df_raw = generate_synthetic_data()

    df              = normalize_and_engineer_features(df_raw)
    df, iqr_report  = remove_outliers_iqr(df)
    events          = detect_events(df)
    summaries       = [summarize_event(ev) for ev in events]
    index           = build_index(events, summaries, llm=llm, embed_model=embed_model)

    return {
        'df':         df,
        'events':     events,
        'summaries':  summaries,
        'index':      index,
        'iqr_report': iqr_report,
    }
