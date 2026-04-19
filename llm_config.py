"""
llm_config.py

Pluggable LLM + embedding backend for SensorSpeak.
Supports Ollama (default, local), OpenAI, and HuggingFace Inference API.

Usage
-----
from llm_config import get_llm, get_embed_model, LLMBackend

llm   = get_llm(LLMBackend.OPENAI)          # needs OPENAI_API_KEY env var
embed = get_embed_model(LLMBackend.OPENAI)   # uses text-embedding-3-small

result = run_pipeline(llm=llm, embed_model=embed)

Environment variables
---------------------
OPENAI_API_KEY          required for OPENAI backend
HF_API_KEY              required for HUGGINGFACE_API backend (free tier works)
OLLAMA_HOST             optional; default http://localhost:11434
SENSOSPEAK_BACKEND      optional; set to 'ollama' | 'openai' | 'hf_api' | 'hf_local'
"""

import os
from enum import Enum
from typing import Optional


class LLMBackend(Enum):
    OLLAMA         = 'ollama'          # local Ollama daemon (default)
    OPENAI         = 'openai'          # OpenAI API  (needs OPENAI_API_KEY)
    HUGGINGFACE_API   = 'hf_api'       # HuggingFace Inference API (needs HF_API_KEY)
    HUGGINGFACE_LOCAL = 'hf_local'     # HuggingFace model downloaded locally (no key)


# ── Default model names per backend ───────────────────────────────────────────
_DEFAULT_LLM_MODELS = {
    LLMBackend.OLLAMA:            'qwen2.5:0.5b',
    LLMBackend.OPENAI:            'gpt-3.5-turbo',
    LLMBackend.HUGGINGFACE_API:   'HuggingFaceH4/zephyr-7b-beta',
    LLMBackend.HUGGINGFACE_LOCAL: 'HuggingFaceH4/zephyr-7b-beta',
}

_DEFAULT_EMBED_MODELS = {
    LLMBackend.OLLAMA:            'BAAI/bge-small-en-v1.5',   # HuggingFace local
    LLMBackend.OPENAI:            'text-embedding-3-small',
    LLMBackend.HUGGINGFACE_API:   'BAAI/bge-small-en-v1.5',
    LLMBackend.HUGGINGFACE_LOCAL: 'BAAI/bge-small-en-v1.5',
}


def get_active_backend() -> LLMBackend:
    """
    Read the SENSOSPEAK_BACKEND environment variable and return the matching enum.
    Falls back to OLLAMA if the variable is unset or unrecognised.
    """
    raw = os.environ.get('SENSOSPEAK_BACKEND', 'ollama').lower().strip()
    mapping = {
        'ollama':   LLMBackend.OLLAMA,
        'openai':   LLMBackend.OPENAI,
        'hf_api':   LLMBackend.HUGGINGFACE_API,
        'hf_local': LLMBackend.HUGGINGFACE_LOCAL,
    }
    if raw not in mapping:
        print(f'[llm_config] Unknown backend "{raw}", falling back to ollama.')
        return LLMBackend.OLLAMA
    return mapping[raw]


def get_llm(backend: Optional[LLMBackend] = None, model: Optional[str] = None):
    """
    Return a LlamaIndex-compatible LLM for the given backend.

    Args:
        backend: LLMBackend enum value. Reads SENSOSPEAK_BACKEND env var if None.
        model:   Model name override. Uses the backend default if None.

    Returns:
        A LlamaIndex BaseLLM instance.

    Raises:
        ImportError  if the required llama-index integration is not installed.
        ValueError   if a required API key environment variable is missing.
    """
    if backend is None:
        backend = get_active_backend()

    model = model or _DEFAULT_LLM_MODELS[backend]

    if backend == LLMBackend.OLLAMA:
        try:
            from llama_index.llms.ollama import Ollama
        except ImportError:
            raise ImportError('Run: pip install llama-index-llms-ollama')
        host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        return Ollama(model=model, base_url=host, request_timeout=180)

    if backend == LLMBackend.OPENAI:
        try:
            from llama_index.llms.openai import OpenAI
        except ImportError:
            raise ImportError('Run: pip install llama-index-llms-openai')
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                'OPENAI_API_KEY environment variable is not set.\n'
                'Get a key at https://platform.openai.com/api-keys\n'
                'Then: export OPENAI_API_KEY=sk-...'
            )
        return OpenAI(model=model, api_key=api_key)

    if backend == LLMBackend.HUGGINGFACE_API:
        try:
            from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
        except ImportError:
            raise ImportError('Run: pip install llama-index-llms-huggingface-api')
        api_key = os.environ.get('HF_API_KEY')
        if not api_key:
            raise ValueError(
                'HF_API_KEY environment variable is not set.\n'
                'Get a free token at https://huggingface.co/settings/tokens\n'
                'Then: export HF_API_KEY=hf_...'
            )
        return HuggingFaceInferenceAPI(model_name=model, token=api_key)

    if backend == LLMBackend.HUGGINGFACE_LOCAL:
        try:
            from llama_index.llms.huggingface import HuggingFaceLLM
        except ImportError:
            raise ImportError('Run: pip install llama-index-llms-huggingface transformers accelerate')
        return HuggingFaceLLM(
            model_name=model,
            tokenizer_name=model,
            max_new_tokens=256,
            # generates_kwargs — temperature kept low for deterministic sensor answers
            generate_kwargs={'temperature': 0.1, 'do_sample': True},
        )

    raise ValueError(f'Unsupported backend: {backend}')


def get_embed_model(backend: Optional[LLMBackend] = None, model: Optional[str] = None):
    """
    Return a LlamaIndex-compatible embedding model for the given backend.

    Ollama and all HuggingFace backends use a local bge-small-en-v1.5 model.
    OpenAI backend uses text-embedding-3-small via the API.
    """
    if backend is None:
        backend = get_active_backend()

    model = model or _DEFAULT_EMBED_MODELS[backend]

    if backend == LLMBackend.OPENAI:
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError:
            raise ImportError('Run: pip install llama-index-embeddings-openai')
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OPENAI_API_KEY environment variable is not set.')
        return OpenAIEmbedding(model=model, api_key=api_key)

    # Ollama and all HuggingFace variants use the local HuggingFace embedding
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError:
        raise ImportError('Run: pip install llama-index-embeddings-huggingface sentence-transformers')
    return HuggingFaceEmbedding(model_name=model)


def describe_backend(backend: Optional[LLMBackend] = None) -> str:
    """Return a human-readable description of the active backend configuration."""
    if backend is None:
        backend = get_active_backend()
    llm_model   = _DEFAULT_LLM_MODELS[backend]
    embed_model = _DEFAULT_EMBED_MODELS[backend]
    needs_key = backend in (LLMBackend.OPENAI, LLMBackend.HUGGINGFACE_API)
    key_var = {
        LLMBackend.OPENAI:          'OPENAI_API_KEY',
        LLMBackend.HUGGINGFACE_API: 'HF_API_KEY',
    }.get(backend, 'none required')
    key_set = os.environ.get(key_var, '') != '' if needs_key else True
    return (
        f'Backend      : {backend.value}\n'
        f'LLM model    : {llm_model}\n'
        f'Embed model  : {embed_model}\n'
        f'API key var  : {key_var}\n'
        f'Key present  : {"YES" if key_set else "NO — set " + key_var}'
    )


if __name__ == '__main__':
    print('Active backend configuration:')
    print(describe_backend())
