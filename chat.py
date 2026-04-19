"""
chat.py — SensorSpeak terminal chat interface

Usage:
    python chat.py              # synthetic data, Ollama LLM
    python chat.py data.csv     # real CSV file
    python chat.py --no-llm     # keyword fallback only (no Ollama needed)
"""

import sys
import argparse

from sensorspeak_core import run_pipeline, query_events
from llm_config import LLMBackend, get_llm, get_embed_model


def main():
    parser = argparse.ArgumentParser(description='SensorSpeak CLI chat')
    parser.add_argument('csv', nargs='?', default=None, help='Path to accelerometer CSV')
    parser.add_argument('--no-llm', action='store_true', help='Skip Ollama; use keyword fallback')
    args = parser.parse_args()

    print('SensorSpeak — local motion event chat')
    print('Running pipeline...', end=' ', flush=True)

    llm = None
    embed_model = None
    if not args.no_llm:
        try:
            llm = get_llm(LLMBackend.OLLAMA)
            embed_model = get_embed_model()
        except Exception as exc:
            print(f'\n[warn] Could not load LLM/embeddings: {exc}')
            print('[warn] Falling back to keyword search.')

    try:
        state = run_pipeline(csv_path=args.csv, llm=llm, embed_model=embed_model)
    except Exception as exc:
        print(f'Pipeline failed: {exc}')
        sys.exit(1)

    events = state['events']
    summaries = state['summaries']
    index = state['index']

    print('done.')
    print(f'Detected {len(events)} event(s): {", ".join(e.event_type for e in events)}')
    print('Type your question, or "quit" to exit.\n')

    while True:
        try:
            question = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye.')
            break

        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            print('Bye.')
            break

        answer = query_events(question, index, summaries, verbose=True)
        print(f'SensorSpeak: {answer}\n')


if __name__ == '__main__':
    main()
