"""
pdf_qa.py — Upload a Bosch sensor datasheet PDF and ask questions.

Combines:
  - Event summaries from the SensorSpeak pipeline
  - Text extracted from one or more PDF datasheets
Into a single VectorStoreIndex so you can query both at once.

Usage
-----
from pdf_qa import build_combined_index, query_combined

# With Ollama running locally:
index = build_combined_index(
    pdf_paths=['BMA400_datasheet.pdf', 'BMI088_datasheet.pdf'],
    events=events,
    summaries=summaries,
)
answer = query_combined(index, 'What is the measurement range of BMA400?')
print(answer)

Colab usage (file upload cell)
-------------------------------
from google.colab import files
uploaded = files.upload()
pdf_paths = list(uploaded.keys())
index = build_combined_index(pdf_paths, events, summaries)
"""

import os
from typing import List, Optional
from sensorspeak_core import (
    MotionEvent,
    OLLAMA_MODEL, OLLAMA_TIMEOUT,
    EMBED_MODEL_NAME, SIMILARITY_TOP_K, RESPONSE_MODE,
    _is_ollama_running, _keyword_fallback,
)


# ── PDF text extraction ────────────────────────────────────────────────────────

def _extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF using pypdf.
    Falls back to a clear error message if pypdf is not installed.
    """
    try:
        import pypdf
    except ImportError:
        raise ImportError(
            'pypdf is not installed. Run: pip install pypdf\n'
            'Then re-run this cell.'
        )

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f'PDF not found: {pdf_path}')

    text_parts = []
    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        n_pages = len(reader.pages)
        print(f'  Reading "{os.path.basename(pdf_path)}" ({n_pages} pages)...')
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ''
            if page_text.strip():
                # Prepend page number so the LLM can cite sources
                text_parts.append(f'[Page {i+1}]\n{page_text.strip()}')

    if not text_parts:
        print(f'  ⚠️  No extractable text found in {os.path.basename(pdf_path)}. '
              f'The PDF may be scanned (image-only). Consider OCR pre-processing.')
        return ''

    return '\n\n'.join(text_parts)


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split long text into overlapping chunks for better retrieval.

    chunk_size: max characters per chunk
    overlap:    characters shared between consecutive chunks (improves context continuity)
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ── Document builders ──────────────────────────────────────────────────────────

def _make_pdf_documents(pdf_paths: List[str], chunk_size: int = 1000):
    """Build LlamaIndex Documents from PDF files with source metadata."""
    try:
        from llama_index.core import Document
    except ImportError:
        raise ImportError('Run: pip install llama-index-core')

    docs = []
    for pdf_path in pdf_paths:
        text = _extract_text_from_pdf(pdf_path)
        if not text:
            continue
        filename = os.path.basename(pdf_path)
        chunks = _chunk_text(text, chunk_size=chunk_size)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                text=chunk,
                metadata={
                    'source':     filename,
                    'chunk':      i + 1,
                    'total_chunks': len(chunks),
                    'doc_type':   'datasheet',
                }
            ))
        print(f'  → {len(chunks)} chunks from "{filename}"')
    return docs


def _make_event_documents(events: List[MotionEvent], summaries: List[str]):
    """Build LlamaIndex Documents from SensorSpeak event summaries."""
    try:
        from llama_index.core import Document
    except ImportError:
        raise ImportError('Run: pip install llama-index-core')

    return [
        Document(
            text=summary,
            metadata={
                'source':     'sensor_events',
                'event_type': ev.type,
                'start':      ev.start,
                'end':        ev.end,
                'max_mag':    round(ev.max_mag, 4),
                'doc_type':   'event',
            }
        )
        for ev, summary in zip(events, summaries)
    ]


# ── Index builder ──────────────────────────────────────────────────────────────

def build_combined_index(
    pdf_paths: List[str],
    events: List[MotionEvent],
    summaries: List[str],
    llm=None,
    embed_model=None,
    chunk_size: int = 1000,
):
    """
    Build a VectorStoreIndex that merges PDF datasheet chunks with event summaries.

    Args:
        pdf_paths:   List of local PDF file paths to ingest.
        events:      MotionEvent list from detect_events().
        summaries:   Plain-English summaries from summarize_event().
        llm:         Optional LlamaIndex LLM override (defaults to Ollama).
        embed_model: Optional embedding model override (defaults to bge-small).
        chunk_size:  Characters per PDF chunk (smaller = more precise retrieval).

    Returns:
        VectorStoreIndex, or None if dependencies are unavailable.
    """
    try:
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        print(f'LlamaIndex not available ({e}). Install with: pip install llama-index-core')
        return None

    # Resolve LLM
    if llm is None:
        if not _is_ollama_running():
            print('Ollama not running — PDF index requires an LLM. '
                  'Start Ollama or pass a custom llm= argument.')
            return None
        llm = Ollama(model=OLLAMA_MODEL, request_timeout=OLLAMA_TIMEOUT)

    # Resolve embedding model
    if embed_model is None:
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    Settings.llm         = llm
    Settings.embed_model = embed_model

    # Build document collections
    print('Building PDF documents...')
    pdf_docs   = _make_pdf_documents(pdf_paths, chunk_size=chunk_size)
    event_docs = _make_event_documents(events, summaries)
    all_docs   = pdf_docs + event_docs

    if not all_docs:
        print('No documents to index. Check that PDF paths are correct and files have extractable text.')
        return None

    print(f'Indexing {len(all_docs)} documents '
          f'({len(pdf_docs)} PDF chunks + {len(event_docs)} event summaries)...')
    index = VectorStoreIndex.from_documents(all_docs)
    print('Combined index ready.')
    return index


# ── Query interface ────────────────────────────────────────────────────────────

def query_combined(index, question: str, verbose: bool = False) -> str:
    """
    Query the combined PDF + event index.

    Falls back to a simple keyword search over summaries if the index is None.
    """
    if not question.strip():
        return 'Please provide a non-empty question.'

    if index is None:
        return (
            'Combined index is not available (Ollama may not be running).\n'
            'Start Ollama and call build_combined_index() again.'
        )

    try:
        qe = index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K,
            response_mode=RESPONSE_MODE,
        )
        response = qe.query(question)
        if verbose:
            print(f'[{len(response.source_nodes)} source nodes retrieved]')
            for node in response.source_nodes:
                src = node.metadata.get('source', 'unknown')
                print(f'  Source: {src}  score={node.score:.3f}')
        return str(response)
    except Exception as exc:
        return f'Query failed: {exc}'


# ── Colab helper ───────────────────────────────────────────────────────────────

def upload_and_build(events: List[MotionEvent], summaries: List[str], **kwargs):
    """
    Interactive helper for Google Colab.
    Opens a file picker, uploads PDFs, builds the combined index.

    Usage in a Colab cell:
        from pdf_qa import upload_and_build
        index = upload_and_build(events, summaries)
    """
    try:
        from google.colab import files as colab_files
    except ImportError:
        raise RuntimeError(
            'upload_and_build() only works inside Google Colab.\n'
            'For local use, call build_combined_index(pdf_paths=[...], events, summaries) directly.'
        )

    print('Select one or more PDF files to upload...')
    uploaded = colab_files.upload()
    if not uploaded:
        print('No files uploaded.')
        return None

    pdf_paths = list(uploaded.keys())
    print(f'Uploaded: {pdf_paths}')
    return build_combined_index(pdf_paths, events, summaries, **kwargs)


# ── Quick demo ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from sensorspeak_core import (
        generate_synthetic_data, normalize_and_engineer_features,
        detect_events, summarize_event,
    )

    df_raw    = generate_synthetic_data()
    df        = normalize_and_engineer_features(df_raw)
    events    = detect_events(df)
    summaries = [summarize_event(ev) for ev in events]

    # Point to a real Bosch PDF to test:
    # index = build_combined_index(['BMA400_datasheet.pdf'], events, summaries)
    # print(query_combined(index, 'What is the operating voltage range?'))

    print('pdf_qa.py loaded. Call build_combined_index() with your PDF paths.')
    print(f'Current events: {[ev.type for ev in events]}')
