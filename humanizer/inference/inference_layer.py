"""
Inference Layer  (Section 3 + 17)
──────────────────────────────────
Loads and manages HuggingFace transformer models.
Personal-use edition — optimised for FLAN-T5-XL on Lightning AI L4.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from humanizer.config import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    MODEL_CACHE_DIR,
    GenerationConfig as GenCfg,
    load_yaml_config,
)

logger = logging.getLogger(__name__)

# In-memory model + tokenizer cache (singleton per process)
_loaded_models: dict[str, tuple[PreTrainedModel, PreTrainedTokenizer]] = {}


def _resolve_device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map config string to torch dtype."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_str, torch.float32)


def load_model(
    model_name: str,
    *,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model + tokeniser from the registry.
    Caches in memory so subsequent calls return the same objects.
    """
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    hf_id = MODEL_REGISTRY.get(model_name, model_name)
    model_cfg = load_yaml_config("model_config.yaml").get("models", {}).get(model_name, {})

    device = device or model_cfg.get("device") or _resolve_device()
    dtype_str = dtype or model_cfg.get("dtype", "float32")
    torch_dtype = _resolve_dtype(dtype_str)

    logger.info("Loading model '%s' (%s) on %s [%s]", model_name, hf_id, device, dtype_str)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=str(MODEL_CACHE_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        hf_id,
        dtype=torch_dtype,
        cache_dir=str(MODEL_CACHE_DIR),
    ).to(device)
    model.eval()

    _loaded_models[model_name] = (model, tokenizer)
    return model, tokenizer


async def generate_text(
    model_name: str,
    input_text: str,
    *,
    gen_cfg: Optional[GenCfg] = None,
    max_new_tokens: Optional[int] = None,
    prefix: str = "paraphrase: ",
    use_sampling: bool = True,
) -> str:
    """
    Run inference on *input_text* using the specified model.

    For T5-family models the *prefix* is prepended (e.g. ``"paraphrase: "``).

    Two decoding strategies:
      - use_sampling=True  → stochastic (temperature + top_p) — varied outputs
      - use_sampling=False → group beam search (deterministic, diverse beams)
    """
    cfg = gen_cfg or GenCfg()
    model, tokenizer = load_model(model_name)
    device = next(model.parameters()).device

    prompt = f"{prefix}{input_text}" if prefix else input_text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    gen_kwargs: dict = {
        **inputs,
        "repetition_penalty": cfg.repetition_penalty,
        "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
        "max_new_tokens": max_new_tokens or cfg.max_new_tokens,
        "max_length": None,   # Prevent model's default max_length=20 from truncating
        "num_return_sequences": 1,
        "trust_remote_code": True,
    }

    if use_sampling:
        # Stochastic sampling — produces varied outputs across runs
        gen_kwargs.update(
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
    else:
        # Group beam search — deterministic, high quality
        gen_kwargs.update(
            num_beams=cfg.num_beams,
            num_beam_groups=cfg.num_beam_groups,
            diversity_penalty=cfg.diversity_penalty,
            do_sample=False,
        )

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def unload_model(model_name: str) -> None:
    """Remove a model from the in-memory cache to free GPU memory."""
    if model_name in _loaded_models:
        del _loaded_models[model_name]
        torch.cuda.empty_cache()
        logger.info("Unloaded model '%s'", model_name)
