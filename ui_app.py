"""
ui_app.py — SensorSpeak Gradio UI  (Apple light mode)

Run:
    python ui_app.py
    python ui_app.py --share
"""

import sys, io
import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from sensorspeak_core import (
    query_events as _query_events, run_pipeline,
    _severity_label, IDLE_STD_MAX, IMPACT_MEAN_MIN,
)
from llm_config import LLMBackend, get_llm, get_embed_model

# ── Pipeline state ──────────────────────────────────────────────────────────────
_state = {'df': None, 'events': [], 'summaries': [], 'index': None, 'iqr': {}}

# ── Apple light palette (hardcoded — no theming) ────────────────────────────────
# bg0 = #f2f2f7   system background
# bg1 = #ffffff   card / panel surface
# bg2 = #f2f2f7   secondary fill (same as bg0)
# bg3 = #e5e5ea   tertiary / hover fill
# sep = rgba(60,60,67,.15)
# lbl1 = #1c1c1e  primary text
# lbl2 = rgba(60,60,67,.55)  secondary text
# lbl3 = rgba(60,60,67,.3)   placeholder / muted
# accent  = #5856d6  system indigo
# accent2 = #007aff  system blue
# green   = #34c759
# red     = #ff3b30
# amber   = #ff9500
# purple  = #af52de
# teal    = #32ade6

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset ───────────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

/* ── Gradio chrome → white / light-gray ─────────────────────────────────────── */
html, body,
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap,
#root {
    background: #f2f2f7 !important;
    color: #1c1c1e !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

footer, .svelte-footer, .footer { display: none !important; }

.gap, .gr-group, .gr-box, .block, .form, .tabitem {
    background: #ffffff !important;
    border-color: rgba(60,60,67,.15) !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────────────── */
input, textarea, select,
[data-testid="textbox"] textarea,
[data-testid="textbox"] input {
    background: #ffffff !important;
    color: #1c1c1e !important;
    border: 1px solid rgba(60,60,67,.2) !important;
    border-radius: 10px !important;
    font-family: inherit !important;
    font-size: .875rem !important;
}
input::placeholder, textarea::placeholder { color: rgba(60,60,67,.3) !important; }
input:focus, textarea:focus {
    border-color: #007aff !important;
    box-shadow: 0 0 0 3px rgba(0,122,255,.12) !important;
    outline: none !important;
}

/* ── Labels ──────────────────────────────────────────────────────────────────── */
label, .label-wrap span, [data-testid="block-label"] span {
    color: rgba(60,60,67,.55) !important;
    font-size: .72rem !important;
    font-weight: 500 !important;
    letter-spacing: .04em !important;
    text-transform: uppercase !important;
}

/* ── Dropdown ────────────────────────────────────────────────────────────────── */
.wrap-inner, [data-testid="dropdown"] .wrap-inner {
    background: #ffffff !important;
    border-color: rgba(60,60,67,.2) !important;
    color: #1c1c1e !important;
}
ul.options { background: #ffffff !important; border-color: rgba(60,60,67,.15) !important; box-shadow: 0 8px 24px rgba(0,0,0,.1) !important; }
ul.options li { color: #1c1c1e !important; }
ul.options li:hover   { background: #f2f2f7 !important; }
ul.options li.selected { background: rgba(88,86,214,.08) !important; color: #5856d6 !important; }

/* ── File upload ─────────────────────────────────────────────────────────────── */
[data-testid="file"] .wrap {
    background: #ffffff !important;
    border: 2px dashed rgba(60,60,67,.2) !important;
    border-radius: 12px !important;
    color: rgba(60,60,67,.55) !important;
}
[data-testid="file"] .wrap:hover { border-color: #5856d6 !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────────── */
button, .gr-button {
    font-family: inherit !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: all .18s cubic-bezier(.4,0,.2,1) !important;
}
button.secondary, .gr-button-secondary {
    background: #ffffff !important;
    border: 1px solid rgba(60,60,67,.2) !important;
    color: rgba(60,60,67,.7) !important;
}
button.secondary:hover {
    background: #f2f2f7 !important;
    border-color: rgba(60,60,67,.35) !important;
    color: #1c1c1e !important;
}

/* ── Chatbot ─────────────────────────────────────────────────────────────────── */
.chatbot, [data-testid="chatbot"], .chatbot .wrap {
    background: #ffffff !important;
    border: 1px solid rgba(60,60,67,.12) !important;
    border-radius: 16px !important;
}
.chatbot .message.user .bubble {
    background: #007aff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 20px 20px 4px 20px !important;
    box-shadow: none !important;
    font-size: .95rem !important;
}
.chatbot .message.bot .bubble,
.chatbot .message.assistant .bubble {
    background: #e9e9eb !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 20px 20px 20px 4px !important;
    box-shadow: none !important;
    font-size: .95rem !important;
}
.chatbot .placeholder { color: rgba(60,60,67,.3) !important; }

/* ── Scrollbars ──────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #e5e5ea; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #c7c7cc; }

/* ═══════════════════════════════════════════════════════════════════════════════ */
/*  SENSORSPEAK COMPONENTS                                                        */
/* ═══════════════════════════════════════════════════════════════════════════════ */

/* ── Hero ────────────────────────────────────────────────────────────────────── */
.ss-hero {
    background: linear-gradient(160deg, #eeeef8 0%, #f0f0fa 50%, #e8eef8 100%);
    border: 1px solid rgba(88,86,214,.15);
    border-radius: 22px;
    padding: 32px 36px 28px;
    margin-bottom: 2px;
    position: relative;
    overflow: hidden;
}
.ss-hero::before {
    content: ''; position: absolute; top: -80px; right: -40px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(88,86,214,.08) 0%, transparent 68%);
    pointer-events: none;
}
.ss-hero::after {
    content: ''; position: absolute; bottom: -60px; left: 15%;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,122,255,.06) 0%, transparent 68%);
    pointer-events: none;
}
.ss-wordmark {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -.05em; line-height: 1;
    color: #1c1c1e; margin-bottom: 6px; position: relative; z-index: 1;
}
.ss-sub {
    font-size: .85rem; color: rgba(60,60,67,.5); font-weight: 400;
    position: relative; z-index: 1;
}
.ss-pills-row { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; position: relative; z-index: 1; }
.ss-pill { padding: 4px 12px; border-radius: 99px; font-size: .68rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; }
.ss-pill-green  { background: rgba(52,199,89,.1);   color: #248a3d; border: 1px solid rgba(52,199,89,.3); }
.ss-pill-indigo { background: rgba(88,86,214,.08);  color: #5856d6; border: 1px solid rgba(88,86,214,.25); }
.ss-pill-blue   { background: rgba(0,122,255,.08);  color: #0071e3; border: 1px solid rgba(0,122,255,.2); }

/* ── Stat cards ──────────────────────────────────────────────────────────────── */
.ss-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 2px;
}
@media (max-width: 700px) { .ss-stats { grid-template-columns: repeat(2, 1fr); } }
.ss-card {
    background: #ffffff;
    border: 1px solid rgba(60,60,67,.12);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
}
.ss-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,.1); }
.ss-card-val { font-size: 2rem; font-weight: 700; letter-spacing: -.05em; line-height: 1; margin-bottom: 5px; }
.ss-card-lbl { font-size: .65rem; font-weight: 500; letter-spacing: .1em; text-transform: uppercase; color: rgba(60,60,67,.4); }
.c-blue   { color: #007aff; }
.c-green  { color: #34c759; }
.c-amber  { color: #ff9500; }
.c-red    { color: #ff3b30; }

/* ── Section title ───────────────────────────────────────────────────────────── */
.ss-sec {
    font-size: .65rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: rgba(60,60,67,.35);
    padding-bottom: 10px; margin-bottom: 14px;
    border-bottom: 1px solid rgba(60,60,67,.1);
}

/* ── Divider ─────────────────────────────────────────────────────────────────── */
.ss-hr { height: 1px; background: rgba(60,60,67,.1); margin: 18px 0; }

/* ── Run button ──────────────────────────────────────────────────────────────── */
#ss-run-btn button {
    background: linear-gradient(135deg, #5856d6 0%, #007aff 100%) !important;
    border: none !important;
    color: #fff !important;
    font-size: .9rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    width: 100% !important;
    box-shadow: 0 2px 12px rgba(88,86,214,.25) !important;
}
#ss-run-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(88,86,214,.4) !important;
    filter: brightness(1.05) !important;
}
#ss-run-btn button:active { transform: translateY(0) !important; }

/* ── Status terminal ─────────────────────────────────────────────────────────── */
#ss-status textarea {
    background: #1c1c1e !important;
    color: #34c759 !important;
    font-family: 'JetBrains Mono', 'Menlo', monospace !important;
    font-size: .78rem !important;
    line-height: 1.65 !important;
    border: 1px solid rgba(52,199,89,.25) !important;
    border-radius: 12px !important;
}

/* ── Event list ──────────────────────────────────────────────────────────────── */
.ss-evl { display: flex; flex-direction: column; gap: 6px; }
.ss-evr {
    display: flex; align-items: center; gap: 10px;
    background: #f9f9fb;
    border: 1px solid rgba(60,60,67,.1);
    border-radius: 12px; padding: 10px 14px; font-size: .8rem;
    transition: box-shadow .15s;
}
.ss-evr:hover { box-shadow: 0 2px 12px rgba(0,0,0,.08); }
.ss-bge {
    font-size: .63rem; font-weight: 700; padding: 3px 10px;
    border-radius: 99px; min-width: 66px; text-align: center;
    font-family: 'JetBrains Mono', monospace; flex-shrink: 0;
}
.b-idle    { background: rgba(52,199,89,.1);  color: #248a3d; border: 1px solid rgba(52,199,89,.3); }
.b-walking { background: rgba(50,173,230,.1); color: #0071e3; border: 1px solid rgba(50,173,230,.3); }
.b-impact  { background: rgba(255,59,48,.1);  color: #d70015; border: 1px solid rgba(255,59,48,.3); }
.b-shaking { background: rgba(255,149,0,.1);  color: #c93400; border: 1px solid rgba(255,149,0,.3); }
.b-unknown { background: rgba(60,60,67,.08);  color: rgba(60,60,67,.5); border: 1px solid rgba(60,60,67,.15); }
.ss-etm { color: rgba(60,60,67,.4); font-size: .7rem; font-family: 'JetBrains Mono', monospace; }
.ss-emg { margin-left: auto; color: rgba(60,60,67,.4); font-size: .7rem; font-family: 'JetBrains Mono', monospace; }
.ss-esv { font-size: .63rem; padding: 2px 8px; border-radius: 99px; flex-shrink: 0; }
.sv-low      { background: rgba(52,199,89,.1);  color: #248a3d; }
.sv-moderate { background: rgba(50,173,230,.1); color: #0071e3; }
.sv-high     { background: rgba(255,149,0,.1);  color: #c93400; }
.sv-severe   { background: rgba(255,59,48,.1);  color: #d70015; }

/* ── IQR block ───────────────────────────────────────────────────────────────── */
.ss-iqr {
    background: #1c1c1e;
    border: 1px solid rgba(88,86,214,.2);
    border-radius: 12px; padding: 14px 16px; margin-top: 14px;
    font-family: 'JetBrains Mono', monospace; font-size: .74rem;
}
.ss-iqr-h {
    color: #a5a3f7; font-weight: 600; font-size: .63rem;
    letter-spacing: .1em; text-transform: uppercase;
    margin-bottom: 10px; padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,.08);
}
.ss-iqr-r { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,.05); }
.ss-iqr-r:last-child { border: none; }
.ss-iqr-k { color: rgba(235,235,245,.35); }
.ss-iqr-v { color: rgba(235,235,245,.75); }
.ss-iqr-v.clip { color: #ff453a !important; font-weight: 600; }

/* ── Question pills ──────────────────────────────────────────────────────────── */
.ss-qps { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 12px; }
.ss-qp {
    font-size: .72rem; font-weight: 500; padding: 5px 13px;
    border-radius: 99px; cursor: pointer; user-select: none;
    background: rgba(88,86,214,.07);
    border: 1px solid rgba(88,86,214,.2);
    color: #5856d6;
    transition: all .15s;
}
.ss-qp:hover { background: rgba(88,86,214,.14); border-color: rgba(88,86,214,.4); color: #3d3ac9; }
.ss-qp:active { transform: scale(.97); }

/* ── Chat send / clear buttons ───────────────────────────────────────────────── */
#ss-send button {
    background: #007aff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 20px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    box-shadow: none !important;
    transition: all .15s !important;
}
#ss-send button:hover { opacity: 0.8 !important; }

#ss-clear button {
    background: transparent !important;
    color: #007aff !important;
    border: none !important;
    font-weight: 500 !important;
    border-radius: 20px !important;
}
#ss-clear button:hover { background: rgba(0,122,255,.1) !important; }

/* ── Chat input ──────────────────────────────────────────────────────────────── */
#ss-qi textarea {
    background: #ffffff !important;
    color: #1c1c1e !important;
    border: 1px solid #c7c7cc !important;
    border-radius: 20px !important;
    font-size: .95rem !important;
    padding: 10px 16px !important;
    resize: none !important;
}
#ss-qi textarea:focus {
    border-color: #007aff !important;
    box-shadow: 0 0 0 1px #007aff !important;
}

/* ── Chart ───────────────────────────────────────────────────────────────────── */
#ss-chart { background: transparent !important; border: none !important; }
#ss-chart img { border-radius: 14px; border: 1px solid rgba(60,60,67,.12); width: 100%; }

/* ── Empty placeholder ───────────────────────────────────────────────────────── */
.ss-mt { color: rgba(60,60,67,.35); font-size: .82rem; font-style: italic; padding: 6px 2px; }
</style>
"""

# ── HTML helpers ────────────────────────────────────────────────────────────────

def _badge_cls(t: str) -> str:
    return f'b-{t}' if t in ('idle', 'walking', 'impact', 'shaking') else 'b-unknown'

def _sev_cls(s: str) -> str:
    return f'sv-{s}'

def _events_html(events: list) -> str:
    if not events:
        return '<p class="ss-mt">No events yet — run the pipeline first.</p>'
    rows = []
    for ev in events:
        sev = _severity_label(ev.max_mag)
        dur = round(ev.end - ev.start, 2)
        rows.append(
            f'<div class="ss-evr">'
            f'<span class="ss-bge {_badge_cls(ev.type)}">{ev.type}</span>'
            f'<span class="ss-etm">{ev.start:.1f}s–{ev.end:.1f}s&nbsp;({dur}s)</span>'
            f'<span class="ss-esv {_sev_cls(sev)}">{sev}</span>'
            f'<span class="ss-emg">↑{ev.max_mag:.3f}</span>'
            f'</div>'
        )
    return f'<div class="ss-evl">{"".join(rows)}</div>'

def _iqr_html(r: dict) -> str:
    if not r:
        return ''
    clp = 'clip' if r.get('pct_clipped', 0) > 0 else ''
    return (
        f'<div class="ss-iqr">'
        f'<div class="ss-iqr-h">◈ IQR Outlier Removal</div>'
        f'<div class="ss-iqr-r"><span class="ss-iqr-k">Q1 / Q3</span><span class="ss-iqr-v">{r["q1"]} / {r["q3"]}</span></div>'
        f'<div class="ss-iqr-r"><span class="ss-iqr-k">IQR</span><span class="ss-iqr-v">{r["iqr"]}</span></div>'
        f'<div class="ss-iqr-r"><span class="ss-iqr-k">Lower fence</span><span class="ss-iqr-v">{r["lower_fence"]}</span></div>'
        f'<div class="ss-iqr-r"><span class="ss-iqr-k">Upper fence</span><span class="ss-iqr-v">{r["upper_fence"]}</span></div>'
        f'<div class="ss-iqr-r"><span class="ss-iqr-k">Clipped</span>'
        f'<span class="ss-iqr-v {clp}">{r["n_clipped"]} ({r["pct_clipped"]}%)</span></div>'
        f'</div>'
    )

def _stats_html(samples='—', events='—', top='—', clipped='—') -> str:
    return (
        f'<div class="ss-stats">'
        f'<div class="ss-card"><div class="ss-card-val c-blue">{samples}</div><div class="ss-card-lbl">Samples</div></div>'
        f'<div class="ss-card"><div class="ss-card-val c-green">{events}</div><div class="ss-card-lbl">Events</div></div>'
        f'<div class="ss-card"><div class="ss-card-val c-amber">{top}</div><div class="ss-card-lbl">Top event</div></div>'
        f'<div class="ss-card"><div class="ss-card-val c-red">{clipped}</div><div class="ss-card-lbl">IQR clipped</div></div>'
        f'</div>'
    )

def _hero_html() -> str:
    return (
        '<div class="ss-hero">'
        '<div class="ss-wordmark">⚡ SensorSpeak</div>'
        '<div class="ss-sub">Bosch Accelerometer · Motion Event Explainer · Natural-Language Query</div>'
        '<div class="ss-pills-row">'
        '<span class="ss-pill ss-pill-green">● Fully Offline</span>'
        '<span class="ss-pill ss-pill-indigo">No API Keys</span>'
        '<span class="ss-pill ss-pill-blue">LlamaIndex + Ollama</span>'
        '</div>'
        '</div>'
    )

def _pills_html() -> str:
    hints = [
        'Were there any dangerous events?',
        'How long was the device idle?',
        'Describe the shaking event.',
        'What happened after 13 seconds?',
        'Summarise all events in order.',
        'What was the peak acceleration?',
    ]
    pills = ''.join(
        f'<span class="ss-qp" '
        f'onclick="var t=document.querySelector(\'#ss-qi textarea\');'
        f't.value=\'{q}\';t.dispatchEvent(new Event(\'input\',{{bubbles:true}}));">'
        f'{q}</span>'
        for q in hints
    )
    return f'<div class="ss-qps">{pills}</div>'

# ── Figure (light theme) ──────────────────────────────────────────────────────────

def _render_figure(df: pd.DataFrame, events: list) -> Image.Image:
    t = df['timestamp']
    BG  = '#ffffff'
    AX  = '#f9f9fb'
    GRD = '#e5e5ea'
    TXT = '#3c3c43'

    ev_pal = {
        'idle':    '#34c759',
        'walking': '#32ade6',
        'impact':  '#ff3b30',
        'shaking': '#ff9500',
        'unknown': '#8e8e93',
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True, facecolor=BG)
    fig.subplots_adjust(hspace=0.08, top=0.93, bottom=0.07, left=0.07, right=0.97)
    fig.suptitle('Signal Overview', color='#1c1c1e', fontsize=11,
                 fontweight='700', x=0.04, ha='left', y=0.97)

    for ax in axes:
        ax.set_facecolor(AX)
        ax.tick_params(colors=TXT, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(GRD)
        ax.grid(color=GRD, linewidth=0.6, alpha=1.0)
        ax.set_axisbelow(True)

    # Panel 1 — raw axes
    axes[0].plot(t, df['_raw_accel_x'], color='#ff3b30', lw=0.8, label='X lateral',  alpha=0.85)
    axes[0].plot(t, df['_raw_accel_y'], color='#34c759', lw=0.8, label='Y forward',  alpha=0.85)
    axes[0].plot(t, df['_raw_accel_z'], color='#007aff', lw=0.8, label='Z vertical', alpha=0.85)
    axes[0].set_ylabel('m/s²', color=TXT, fontsize=8)
    axes[0].legend(loc='upper right', fontsize=7, facecolor=BG,
                   edgecolor=GRD, labelcolor='#1c1c1e', framealpha=0.95)

    # Panel 2 — magnitude + event bands
    for ev in events:
        axes[1].axvspan(ev.start, ev.end, alpha=0.15,
                        color=ev_pal.get(ev.type, '#8e8e93'), zorder=1)
    axes[1].plot(t, df['accel_magnitude'], color='#af52de', lw=1.0, label='Magnitude', zorder=2)
    if '_iqr_outlier' in df.columns:
        out = df[df['_iqr_outlier']]
        if len(out):
            axes[1].scatter(out['timestamp'], out['accel_magnitude'],
                            c='#ff3b30', s=14, zorder=5, label='IQR clipped', alpha=0.9)
    patches = [mpatches.Patch(color=c, alpha=0.6, label=k) for k, c in ev_pal.items()]
    mag_h   = plt.Line2D([0], [0], color='#af52de', lw=1.5, label='Magnitude')
    axes[1].legend(handles=patches + [mag_h], loc='upper right', fontsize=6.5,
                   facecolor=BG, edgecolor=GRD, labelcolor='#1c1c1e',
                   framealpha=0.95, ncol=3)
    axes[1].set_ylabel('Magnitude', color=TXT, fontsize=8)

    # Panel 3 — rolling stats + thresholds
    axes[2].plot(t, df['rolling_mean'], color='#ff9500', lw=1.0, label='Rolling mean')
    axes[2].plot(t, df['rolling_std'],  color='#5856d6', lw=1.0, label='Rolling std')
    axes[2].axhline(IDLE_STD_MAX,    color='#34c759', ls='--', lw=0.9, alpha=0.8,
                    label=f'Idle ≤ {IDLE_STD_MAX}')
    axes[2].axhline(IMPACT_MEAN_MIN, color='#ff3b30', ls='--', lw=0.9, alpha=0.8,
                    label=f'Impact ≥ {IMPACT_MEAN_MIN}')
    axes[2].legend(loc='upper right', fontsize=7, facecolor=BG,
                   edgecolor=GRD, labelcolor='#1c1c1e', framealpha=0.95, ncol=2)
    axes[2].set_ylabel('Value', color=TXT, fontsize=8)
    axes[2].set_xlabel('Time (s)', color=TXT, fontsize=8.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ── Pipeline handler ─────────────────────────────────────────────────────────────

BACKEND_MAP = {
    'Ollama — local, no key (default)':   LLMBackend.OLLAMA,
    'OpenAI GPT — needs OPENAI_API_KEY':  LLMBackend.OPENAI,
    'HuggingFace API — needs HF_API_KEY': LLMBackend.HUGGINGFACE_API,
    'HuggingFace Local — no key':         LLMBackend.HUGGINGFACE_LOCAL,
}

def run(csv_file, backend_name: str):
    backend = BACKEND_MAP.get(backend_name, LLMBackend.OLLAMA)
    warn = ''
    try:
        llm         = get_llm(backend)
        embed_model = get_embed_model(backend)
    except (ImportError, ValueError) as e:
        llm = embed_model = None
        warn = f'⚠  {e}\n→  Keyword fallback active.\n\n'

    try:
        result = run_pipeline(
            csv_path=csv_file if csv_file is not None else None,
            llm=llm, embed_model=embed_model,
        )
    except ValueError as e:
        return (f'❌  {e}', '<p class="ss-mt">Pipeline error.</p>',
                '', _stats_html(), '', None)

    _state.update(result)
    df     = result['df']
    events = result['events']
    iqr    = result['iqr_report']
    src    = 'uploaded CSV' if csv_file else 'synthetic data'
    idxtag = 'LlamaIndex + LLM' if result['index'] else 'keyword fallback'

    status = (
        f'{warn}'
        f'✅  Pipeline ready\n'
        f'    Samples    : {len(df):,}\n'
        f'    Duration   : {df["timestamp"].iloc[-1]:.1f} s @ 100 Hz\n'
        f'    Events     : {len(events)}\n'
        f'    IQR clipped: {iqr["n_clipped"]} ({iqr["pct_clipped"]}%)\n'
        f'    Backend    : {backend.value}  [{idxtag}]\n'
        f'    Source     : {src}'
    )

    counts   = {}
    for ev in events:
        counts[ev.type] = counts.get(ev.type, 0) + 1
    dominant = max(counts, key=counts.get) if counts else '—'

    return (
        status,
        _events_html(events),
        _iqr_html(iqr),
        _stats_html(f'{len(df):,}', str(len(events)), dominant, str(iqr['n_clipped'])),
        '',
        _render_figure(df, events),
    )

# ── Chat handler ──────────────────────────────────────────────────────────────────

def chat(user_msg: str, history: list):
    if not user_msg.strip():
        return '', history
    if _state['df'] is None:
        bot = '⚠️  Run the pipeline first — click **▶ Run Pipeline** on the left.'
    else:
        bot = _query_events(user_msg, index=_state['index'],
                            summaries=_state['summaries'])
    history.append({'role': 'user',      'content': user_msg})
    history.append({'role': 'assistant', 'content': bot})
    return '', history

def clear_chat():
    return []

# ── UI builder ────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title='SensorSpeak') as demo:

        gr.HTML(STYLE)
        gr.HTML(_hero_html())

        stats_html = gr.HTML(_stats_html())

        with gr.Row(equal_height=False):

            # ── Left: config + events ──────────────────────────────────────
            with gr.Column(scale=5, min_width=280):

                gr.HTML('<div class="ss-sec">Configuration</div>')

                csv_upload = gr.File(
                    label='Sensor CSV  (leave empty → synthetic data)',
                    file_types=['.csv'], type='filepath',
                )
                backend_dd = gr.Dropdown(
                    choices=list(BACKEND_MAP.keys()),
                    value='Ollama — local, no key (default)',
                    label='LLM Backend',
                )
                run_btn = gr.Button(
                    '▶  Run Pipeline', variant='primary',
                    size='lg', elem_id='ss-run-btn',
                )
                status_box = gr.Textbox(
                    label='Pipeline Status', lines=7,
                    interactive=False, elem_id='ss-status',
                )

                gr.HTML('<div class="ss-hr"></div>')
                gr.HTML('<div class="ss-sec">Detected Events</div>')

                events_display = gr.HTML('<p class="ss-mt">Run the pipeline to see events.</p>')
                iqr_display    = gr.HTML('')

            # ── Right: chat + chart ────────────────────────────────────────
            with gr.Column(scale=9):

                gr.HTML('<div class="ss-sec">Ask Questions About Your Data</div>')
                gr.HTML(_pills_html())

                chatbot = gr.Chatbot(
                    height=320, show_label=False,
                    placeholder='**Run the pipeline**, then ask a question about your sensor data.',
                )

                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder='e.g.  Were there any dangerous events?',
                        label='', show_label=False, scale=7, elem_id='ss-qi',
                    )
                    send_btn  = gr.Button('Send',  variant='primary',   scale=1, elem_id='ss-send')
                    clear_btn = gr.Button('Clear', variant='secondary', scale=1, elem_id='ss-clear')

                gr.HTML('<div class="ss-hr"></div>')
                gr.HTML('<div class="ss-sec">Signal Overview</div>')

                figure_out = gr.Image(
                    label='', show_label=False, type='pil', elem_id='ss-chart',
                )

        # ── Event wiring ──────────────────────────────────────────────────
        run_btn.click(
            fn=run,
            inputs=[csv_upload, backend_dd],
            outputs=[status_box, events_display, iqr_display,
                     stats_html, question_box, figure_out],
        )
        send_btn.click(
            fn=chat, inputs=[question_box, chatbot],
            outputs=[question_box, chatbot],
        )
        question_box.submit(
            fn=chat, inputs=[question_box, chatbot],
            outputs=[question_box, chatbot],
        )
        clear_btn.click(fn=clear_chat, outputs=[chatbot])

    return demo


if __name__ == '__main__':
    app = build_ui()
    app.launch(
        share='--share' in sys.argv,
        server_port=7860,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.gray,
        ),
    )
