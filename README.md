# Humanizer v2.0 — Text Rewriting System

> Transformer-based text rewriting system that produces semantically faithful, structurally original rewrites **undetectable as AI-generated**.

## Architecture

```
Understand → Reconstruct → Generate → Humanize → Validate
```

**Pipeline layers:** Preprocessing → Model Router → Discourse Planner → Encoder → Multi-Style Sampler → Multi-Pass Decoder → Critic → Human Noise Injection → Validation (8 axes) → Meta-Rewriter → Post-Processing

## Quick Start

### 1. Install dependencies

```bash
# On Lightning AI / any GPU environment
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

### 2. Run the API

```bash
uvicorn humanizer.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test a rewrite

```bash
curl -X POST http://localhost:8000/api/v1/rewrite \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog.", "quality_tier": "standard"}'
```

### 4. Run tests

```bash
pytest tests/ -v
```

## Project Structure

```
humanizer/
├── api/            # FastAPI app, routes, schemas
├── preprocessing/  # PII masking, language detection, abuse filtering
├── router/         # Token count × quality tier model selection
├── inference/      # HuggingFace model loading + generation
├── validation/     # 8-axis validation: semantic, NLI, entities, safety
├── postprocessing/ # PII restoration, confidence scoring
├── pipeline/       # Multi-pass generation (Phase 2)
├── humanize/       # Noise injection, perplexity control (Phase 3)
├── meta_rewriter/  # Cross-model fingerprint break (Phase 4)
├── training/       # SFT, RLHF, adversarial training (Phase 5)
└── evaluation/     # Offline eval pipeline (Phase 6)
```

## Deployment

**Development:** Lightning AI Studio (free GPU tier)

**Production (free options):**
- [Hugging Face Spaces](https://huggingface.co/spaces) — free GPU via Gradio/FastAPI
- [Render](https://render.com) — free web service tier
- [Railway](https://railway.app) — $5 free credit/month
- [GitHub Codespaces](https://github.com/codespaces) — dev environment

## Tech Stack

| Component | Tool |
|-----------|------|
| API | FastAPI + Uvicorn |
| Models | HuggingFace Transformers (T5, BART, FLAN, LLaMA) |
| Embeddings | Sentence-Transformers (SBERT) |
| NLI | DeBERTa cross-encoder |
| NER | spaCy |
| Toxicity | Detoxify |
| Perplexity | KenLM / GPT-2 scorer |

## License

MIT
