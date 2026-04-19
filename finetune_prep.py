"""
finetune_prep.py — Convert SensorSpeak synthetic data into a fine-tuning dataset.

WHAT THIS DOES
--------------
Generates instruction-response pairs in JSONL format from the synthetic pipeline.
The resulting dataset can be used to fine-tune a small HuggingFace model (e.g.
Qwen2.5-0.5B, SmolLM2-360M) with HuggingFace PEFT + LoRA so it becomes more
accurate at answering sensor-domain questions.

OUTPUT FORMAT (Alpaca-style JSONL)
----------------------------------
Each line is one JSON object:
{
  "instruction": "Classify this motion reading: rolling_mean=1.4, rolling_std=0.6",
  "input":       "",
  "output":      "This is a walking event. rolling_mean is in the 0.8-2.5 range ..."
}

FINE-TUNING OPTIONS
-------------------
1. HuggingFace PEFT + LoRA (recommended for Colab T4):
   - Works with any model on HuggingFace Hub
   - ~30 min on T4 for 500-sample dataset
   - See the printed instructions at the bottom

2. Ollama Modelfile (no training, just system prompt injection):
   - Creates a custom model variant with built-in sensor knowledge
   - Zero GPU required
   - Useful for small improvements without real fine-tuning

3. HuggingFace AutoTrain (no-code):
   - Upload the JSONL to https://ui.autotrain.huggingface.co
   - Select model, click Train — returns a fine-tuned model card

RUN
---
    python finetune_prep.py
    # Writes: outputs/finetune_dataset.jsonl
    # Prints: fine-tuning command for HuggingFace PEFT
"""

import json
import os
import random
from typing import List

from sensorspeak_core import (
    generate_synthetic_data,
    normalize_and_engineer_features,
    detect_events,
    summarize_event,
    MotionEvent,
    _classify_sample,
    _severity_label,
    IDLE_STD_MAX, IMPACT_MEAN_MIN, IMPACT_STD_MIN,
    SHAKING_STD_MIN, WALKING_MEAN_MIN, WALKING_MEAN_MAX,
    WALKING_STD_MIN, WALKING_STD_MAX,
)

OUTPUT_JSONL = os.path.join('outputs', 'finetune_dataset.jsonl')

# ── Sample generators ──────────────────────────────────────────────────────────

def _classification_samples(n: int = 200) -> List[dict]:
    """
    Generate N classification instruction-response pairs.
    Covers all five event types with varied values.
    """
    import numpy as np
    rng = random.Random(42)
    samples = []

    # Pre-defined (rmean, rstd, expected_type) seeds that hit each class
    class_seeds = [
        (0.02, 0.03, 'idle'),
        (0.10, 0.08, 'idle'),
        (1.2,  0.4,  'walking'),
        (2.0,  0.9,  'walking'),
        (2.8,  1.5,  'impact'),
        (3.5,  2.0,  'impact'),
        (1.0,  2.0,  'shaking'),
        (0.5,  1.9,  'shaking'),
        (0.4,  0.5,  'unknown'),
        (2.4,  0.05, 'unknown'),
    ]

    for _ in range(n):
        seed_rmean, seed_rstd, expected = rng.choice(class_seeds)
        # Add small jitter so each sample is unique
        rmean = max(0.0, seed_rmean + rng.gauss(0, 0.05))
        rstd  = max(0.0, seed_rstd  + rng.gauss(0, 0.03))
        label = _classify_sample(rmean, rstd)   # ground truth from the rule engine
        sev   = _severity_label(rmean + rstd)

        instruction = (
            f'Classify this accelerometer reading and explain why:\n'
            f'rolling_mean = {rmean:.3f}, rolling_std = {rstd:.3f}'
        )
        output = (
            f'This reading is classified as "{label}" ({sev} severity).\n'
            f'Reason: '
        )
        if label == 'idle':
            output += (f'rolling_std ({rstd:.3f}) is below the idle threshold '
                       f'({IDLE_STD_MAX}), indicating minimal vibration and a stationary device.')
        elif label == 'impact':
            output += (f'rolling_mean ({rmean:.3f}) exceeds {IMPACT_MEAN_MIN} AND '
                       f'rolling_std ({rstd:.3f}) exceeds {IMPACT_STD_MIN}, '
                       f'indicating a sharp transient spike consistent with a drop or collision.')
        elif label == 'shaking':
            output += (f'rolling_std ({rstd:.3f}) exceeds the shaking threshold '
                       f'({SHAKING_STD_MIN}), indicating rapid sustained oscillation.')
        elif label == 'walking':
            output += (f'rolling_mean ({rmean:.3f}) is in [{WALKING_MEAN_MIN}, {WALKING_MEAN_MAX}] '
                       f'and rolling_std ({rstd:.3f}) is in [{WALKING_STD_MIN}, {WALKING_STD_MAX}], '
                       f'consistent with rhythmic periodic motion such as footsteps.')
        else:
            output += (f'The values do not match any defined rule. '
                       f'rolling_mean={rmean:.3f} and rolling_std={rstd:.3f} '
                       f'fall outside all detection thresholds. Threshold tuning may be needed.')

        samples.append({'instruction': instruction, 'input': '', 'output': output})

    return samples


def _summary_samples(events: List[MotionEvent], summaries: List[str]) -> List[dict]:
    """
    Generate question-answer pairs from detected events.
    Each event produces 3 different question phrasings.
    """
    samples = []
    question_templates = [
        'What happened between {start:.1f}s and {end:.1f}s?',
        'Describe the motion event that occurred at {start:.1f} seconds.',
        'Was there any notable activity around {mid:.1f} seconds?',
    ]
    for ev, summary in zip(events, summaries):
        mid = (ev.start + ev.end) / 2
        for tmpl in question_templates:
            q = tmpl.format(start=ev.start, end=ev.end, mid=mid)
            samples.append({'instruction': q, 'input': '', 'output': summary})
    return samples


def _threshold_qa_samples() -> List[dict]:
    """
    Generate Q&A pairs about detection thresholds — builds domain knowledge into the model.
    """
    return [
        {
            'instruction': 'What rolling_std threshold is used to detect an idle period?',
            'input': '',
            'output': (f'A sample is classified as idle when rolling_std is below '
                       f'{IDLE_STD_MAX} m/s². This threshold indicates minimal vibration, '
                       f'consistent with a device at rest.'),
        },
        {
            'instruction': 'How is an impact event detected in the SensorSpeak pipeline?',
            'input': '',
            'output': (f'An impact event is detected when rolling_mean exceeds {IMPACT_MEAN_MIN} '
                       f'AND rolling_std exceeds {IMPACT_STD_MIN}. Both conditions must be true '
                       f'simultaneously, ensuring that only sudden high-amplitude transients '
                       f'(drops, collisions, hammer blows) are flagged.'),
        },
        {
            'instruction': 'What distinguishes a shaking event from a walking event?',
            'input': '',
            'output': (f'Shaking is flagged when rolling_std > {SHAKING_STD_MIN}, regardless '
                       f'of rolling_mean. Walking requires rolling_mean in '
                       f'[{WALKING_MEAN_MIN}, {WALKING_MEAN_MAX}] AND rolling_std in '
                       f'[{WALKING_STD_MIN}, {WALKING_STD_MAX}]. Shaking has higher variability '
                       f'and higher frequency oscillation than the rhythmic pattern of walking.'),
        },
        {
            'instruction': 'What is the minimum number of consecutive samples needed to register an event?',
            'input': '',
            'output': 'A run must have at least 10 consecutive same-label samples to register as an event. '
                      'This prevents very short noise spikes from creating false events.',
        },
        {
            'instruction': 'Why does the Z axis have a baseline of 9.81 m/s²?',
            'input': '',
            'output': 'Gravity pulls the sensor downward with 9.81 m/s². '
                      'When the device is flat and stationary, the Z axis measures this constant '
                      'gravitational acceleration. The pipeline uses z-score normalisation so that '
                      'this offset does not distort the magnitude and rolling statistics.',
        },
    ]


def build_dataset() -> List[dict]:
    """Build the full fine-tuning dataset."""
    df_raw    = generate_synthetic_data()
    df        = normalize_and_engineer_features(df_raw)
    events    = detect_events(df)
    summaries = [summarize_event(ev) for ev in events]

    dataset = (
        _classification_samples(n=200) +
        _summary_samples(events, summaries) +
        _threshold_qa_samples()
    )

    # Shuffle so all types are distributed through the file
    random.Random(42).shuffle(dataset)
    return dataset


def save_dataset(dataset: List[dict], path: str = OUTPUT_JSONL) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'Saved {len(dataset)} samples → {path}')


def print_finetune_instructions(jsonl_path: str) -> None:
    """Print copy-paste fine-tuning commands for each method."""
    print('\n' + '=' * 65)
    print('FINE-TUNING OPTIONS')
    print('=' * 65)

    print("""
─── OPTION 1: HuggingFace PEFT + LoRA (Colab T4, ~30 min) ──────────

1. Install dependencies:
   pip install peft transformers trl datasets accelerate bitsandbytes

2. Fine-tune with TRL SFTTrainer (paste into a Colab cell):

   from datasets import load_dataset
   from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
   from peft import LoraConfig, get_peft_model
   from trl import SFTTrainer, SFTConfig
   import torch

   MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'  # or 'HuggingFaceTB/SmolLM2-360M-Instruct'

   tokenizer = AutoTokenizer.from_pretrained(MODEL)
   model = AutoModelForCausalLM.from_pretrained(
       MODEL,
       quantization_config=BitsAndBytesConfig(load_in_4bit=True),
       device_map='auto',
   )
   lora_config = LoraConfig(r=8, lora_alpha=16, target_modules='all-linear',
                             lora_dropout=0.05, task_type='CAUSAL_LM')
   model = get_peft_model(model, lora_config)

   dataset = load_dataset('json', data_files='""" + jsonl_path + """', split='train')

   def format_sample(ex):
       return {'text': f"### Instruction:\\n{ex['instruction']}\\n\\n### Response:\\n{ex['output']}"}

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

3. Use the fine-tuned model in SensorSpeak:
   export SENSOSPEAK_BACKEND=hf_local
   # Then in llm_config.py set model='./sensospeak_lora'
""")

    print("""
─── OPTION 2: Ollama Modelfile (no training, no GPU) ────────────────

Create a file named 'Modelfile' with this content:

   FROM qwen2.5:0.5b
   SYSTEM \"\"\"
   You are SensorSpeak, an expert at interpreting Bosch accelerometer data.
   You understand motion events: idle (rolling_std < 0.15), walking (mean 0.8-2.5,
   std 0.10-1.8), impact (mean > 2.5 AND std > 1.0), shaking (std > 1.8).
   Always cite the time range, severity, and physical explanation in your answers.
   \"\"\"

Then register it with Ollama:
   ollama create sensospeak -f Modelfile
   ollama run sensospeak

And update OLLAMA_MODEL in sensorspeak_core.py:
   OLLAMA_MODEL = 'sensospeak'
""")

    print("""
─── OPTION 3: HuggingFace AutoTrain (no-code, web UI) ───────────────

1. Go to https://ui.autotrain.huggingface.co
2. Create new project → Text Generation → Upload """ + jsonl_path + """
3. Select model: Qwen/Qwen2.5-0.5B-Instruct
4. Set instruction column = 'instruction', response column = 'output'
5. Click Train — HuggingFace runs it in their cloud
6. Download the model and use with llm_config.py HUGGINGFACE_LOCAL backend
""")
    print('=' * 65)


if __name__ == '__main__':
    print('Building fine-tuning dataset from synthetic pipeline data...')
    dataset = build_dataset()
    save_dataset(dataset)
    print_finetune_instructions(OUTPUT_JSONL)
