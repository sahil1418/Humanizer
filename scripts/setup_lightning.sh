#!/bin/bash
# ─────────────────────────────────────────────────────────
# Humanizer v2.0 — Lightning AI Setup Script
# ─────────────────────────────────────────────────────────
# Run this once when you first open the Lightning AI Studio.
#
# Usage:
#   chmod +x scripts/setup_lightning.sh
#   bash scripts/setup_lightning.sh
# ─────────────────────────────────────────────────────────

set -e

echo "═══════════════════════════════════════════════════"
echo "  Humanizer v2.0 — Lightning AI Setup"
echo "═══════════════════════════════════════════════════"

# 1. Install the package + dev dependencies
echo ""
echo "▸ Installing humanizer package..."
pip install -e ".[dev]" --quiet

# 2. Download spaCy model
echo "▸ Downloading spaCy English model..."
python -m spacy download en_core_web_sm --quiet

# 3. Pre-download the models we need for validation
echo "▸ Pre-downloading models (this may take a few minutes)..."
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

models = [
    ('google/flan-t5-base', 'FLAN-T5-base (instruction-tuned, better for rewriting)'),
]

for model_id, desc in models:
    print(f'  Downloading {desc}...')
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForSeq2SeqLM.from_pretrained(model_id)
    print(f'  ✓ {model_id} ready')

print('All models downloaded.')
"

# 4. Run unit tests
echo ""
echo "▸ Running unit tests..."
python -m pytest tests/ -v --tb=short

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ Setup complete!"
echo ""
echo "  Next: Run the validation script:"
echo "    python scripts/validate_pipeline.py"
echo ""
echo "  Or start the API:"
echo "    uvicorn humanizer.api.app:app --host 0.0.0.0 --port 8000"
echo "═══════════════════════════════════════════════════"
