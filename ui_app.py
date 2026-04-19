"""
ui_app.py — SensorSpeak Gradio Chat UI

Run locally:
    python ui_app.py

Run in Colab (paste into a cell):
    !python ui_app.py &
    import time; time.sleep(3)

Then open the localhost URL (or the public share link in Colab).
"""

import os
import sys
import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from PIL import Image

from sensorspeak_core import (
    generate_synthetic_data,
    normalize_and_engineer_features,
    detect_events,
    summarize_event,
    build_index,
    query_events as _query_events,
    run_pipeline,
    MotionEvent,
    EVENT_COLORS,
    VIZ_OUTPUT_FILE,
    ROLLING_WINDOW,
    IDLE_STD_MAX,
    IMPACT_MEAN_MIN,
)
from llm_config import LLMBackend, get_llm, get_embed_model, describe_backend

# ── Global pipeline state ──────────────────────────────────────────────────────
# Re-initialised whenever the user changes the CSV or backend selection.
_state = {
    'df':        None,
    'events':    [],
    'summaries': [],
    'index':     None,
    'history':   [],   # list of (user_msg, bot_msg) tuples for the chatbot
}


def _build_events_table(events):
    """Format detected events as a markdown table string."""
    if not events:
        return '*No events detected yet.*'
    rows = ['| # | Type | Start (s) | End (s) | Duration (s) | Max Mag | Severity |',
            '|---|---|---|---|---|---|---|']
    for i, ev in enumerate(events, 1):
        from sensorspeak_core import _severity_label
        sev = _severity_label(ev.max_mag)
        rows.append(
            f'| {i} | **{ev.type}** | {ev.start:.2f} | {ev.end:.2f} | '
            f'{ev.end - ev.start:.2f} | {ev.max_mag:.4f} | {sev} |'
        )
    return '\n'.join(rows)


def _render_figure(df, events) -> Image.Image:
    """Render the 3-panel figure and return a PIL Image."""
    t = df['timestamp']
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('SensorSpeak — Accelerometer Overview', fontsize=13, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(t, df['_raw_accel_x'], color='#E74C3C', lw=0.7, label='X')
    ax1.plot(t, df['_raw_accel_y'], color='#2ECC71', lw=0.7, label='Y')
    ax1.plot(t, df['_raw_accel_z'], color='#3498DB', lw=0.7, label='Z')
    ax1.set_ylabel('m/s²')
    ax1.set_title('Raw Axes')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    for ev in events:
        ax2.axvspan(ev.start, ev.end, alpha=0.25,
                    color=EVENT_COLORS.get(ev.type, '#D3D3D3'))
    ax2.plot(t, df['accel_magnitude'], color='#8E44AD', lw=0.8, label='Magnitude')
    patches = [mpatches.Patch(color=c, alpha=0.5, label=k) for k, c in EVENT_COLORS.items()]
    ax2.legend(handles=patches + [plt.Line2D([0],[0], color='#8E44AD', lw=1.5, label='Mag')],
               fontsize=7, loc='upper right', ncol=3)
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Magnitude + Event Bands')
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.plot(t, df['rolling_mean'], color='#F39C12', lw=1.0, label='Rolling mean')
    ax3.plot(t, df['rolling_std'],  color='#1ABC9C', lw=1.0, label='Rolling std')
    ax3.axhline(IDLE_STD_MAX,    color='#90EE90', ls='--', lw=0.8, alpha=0.7)
    ax3.axhline(IMPACT_MEAN_MIN, color='#FF6B6B', ls='--', lw=0.8, alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Value')
    ax3.set_title('Rolling Statistics')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def initialise_pipeline(csv_file, backend_name: str):
    """
    Load data, run pipeline, build index.
    Called on app start and whenever the user uploads a new CSV or changes backend.
    Returns: (status_msg, events_table_md, figure_image)
    """
    backend_map = {
        'Ollama (local, no key)':          LLMBackend.OLLAMA,
        'OpenAI (needs OPENAI_API_KEY)':   LLMBackend.OPENAI,
        'HuggingFace API (needs HF_API_KEY)': LLMBackend.HUGGINGFACE_API,
        'HuggingFace Local (no key)':      LLMBackend.HUGGINGFACE_LOCAL,
    }
    backend = backend_map.get(backend_name, LLMBackend.OLLAMA)

    try:
        llm         = get_llm(backend)
        embed_model = get_embed_model(backend)
    except (ImportError, ValueError) as e:
        llm = embed_model = None
        status_warn = f'⚠️  Backend warning: {e}\n→ Falling back to keyword search.\n\n'
    else:
        status_warn = ''

    csv_path = csv_file.name if csv_file is not None else None

    try:
        result = run_pipeline(csv_path=csv_path, llm=llm, embed_model=embed_model)
    except ValueError as e:
        return f'❌ Error: {e}', '*No events.*', None

    _state['df']        = result['df']
    _state['events']    = result['events']
    _state['summaries'] = result['summaries']
    _state['index']     = result['index']
    _state['history']   = []

    n_events = len(result['events'])
    index_type = 'LlamaIndex + LLM' if result['index'] else 'Keyword fallback'
    status = (
        f'{status_warn}'
        f'✅ Pipeline ready\n'
        f'  Samples : {len(result["df"]):,}\n'
        f'  Events  : {n_events}\n'
        f'  Backend : {backend.value}  ({index_type})\n'
        f'  Source  : {"uploaded CSV" if csv_path else "synthetic data"}'
    )
    table = _build_events_table(result['events'])
    fig   = _render_figure(result['df'], result['events'])
    return status, table, fig


def chat(user_message: str, history: list):
    """
    Gradio chat handler.
    history is a list of [user_msg, bot_msg] pairs (Gradio format).
    """
    if _state['df'] is None:
        bot = '⚠️ Pipeline not initialised. Click **Run Pipeline** first.'
        history.append([user_message, bot])
        return '', history

    answer = _query_events(
        user_message,
        index=_state['index'],
        summaries=_state['summaries'],
    )
    history.append([user_message, answer])
    return '', history


def clear_chat():
    _state['history'] = []
    return []


# ── Gradio UI layout ───────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title='SensorSpeak', theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            '# SensorSpeak — Bosch Accelerometer Motion Event Explainer\n'
            'Upload a sensor CSV (or use synthetic data) · Run the pipeline · Ask questions in natural language.'
        )

        with gr.Row():

            # ── Left column: controls ────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown('### ⚙️ Configuration')

                csv_upload = gr.File(
                    label='Upload Sensor CSV (optional)',
                    file_types=['.csv'],
                    type='filepath',
                )

                backend_selector = gr.Dropdown(
                    choices=[
                        'Ollama (local, no key)',
                        'OpenAI (needs OPENAI_API_KEY)',
                        'HuggingFace API (needs HF_API_KEY)',
                        'HuggingFace Local (no key)',
                    ],
                    value='Ollama (local, no key)',
                    label='LLM Backend',
                )

                run_btn = gr.Button('▶ Run Pipeline', variant='primary')
                status_box = gr.Textbox(
                    label='Pipeline Status',
                    lines=6,
                    interactive=False,
                )

                gr.Markdown('### 📋 Detected Events')
                events_table = gr.Markdown('*Click Run Pipeline to load.*')

            # ── Right column: chat + figure ───────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown('### 💬 Ask Questions About Your Sensor Data')

                chatbot = gr.Chatbot(
                    label='SensorSpeak Chat',
                    height=380,
                    show_copy_button=True,
                )

                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder='e.g. Were there any high-severity events?',
                        label='Your question',
                        scale=4,
                    )
                    send_btn = gr.Button('Send', variant='primary', scale=1)

                clear_btn = gr.Button('Clear chat', size='sm')

                gr.Markdown(
                    '**Example questions:**  \n'
                    '`Were there any abnormal events?` · '
                    '`How long was the device idle?` · '
                    '`Describe any impact events.` · '
                    '`What happened after 10 seconds?`'
                )

                gr.Markdown('### 📊 Signal Overview')
                figure_display = gr.Image(label='Accelerometer Plot', type='pil')

        # ── Wire up events ────────────────────────────────────────────────────
        run_btn.click(
            fn=initialise_pipeline,
            inputs=[csv_upload, backend_selector],
            outputs=[status_box, events_table, figure_display],
        )

        send_btn.click(
            fn=chat,
            inputs=[question_box, chatbot],
            outputs=[question_box, chatbot],
        )
        question_box.submit(
            fn=chat,
            inputs=[question_box, chatbot],
            outputs=[question_box, chatbot],
        )
        clear_btn.click(fn=clear_chat, outputs=[chatbot])

    return demo


if __name__ == '__main__':
    app = build_ui()
    # share=True creates a public Gradio URL — useful in Colab
    app.launch(share='--share' in sys.argv, server_port=7860)
