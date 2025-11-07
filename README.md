# 🎬 Movie Review Sentiment Analysis

A fast, lightweight **movie review sentiment** web app built with **Streamlit** and **Hugging Face Transformers**.  
It runs **DistilBERT SST-2** (English) locally, adds **token-level explanations**, **aspect hints**, and a full **evaluation suite** (Confusion Matrix, ROC, PR, Calibration, Threshold sweep, and slice metrics). Figures are sized for a clean, compact UI.

<p align="left">
  <a href="https://python.org">Python</a> •
  <a href="https://streamlit.io/">Streamlit</a> •
  <a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">Transformers</a>
</p>

## ✨ Highlights
- **English-only** model: `distilbert-base-uncased-finetuned-sst-2-english`
- **Single review analysis**: prediction, confidence, optional token influence + aspect hints
- **Evaluation tab**: Confusion Matrix, ROC, PR, Calibration (ECE), threshold sweep, **slice metrics by text length**
- **Compact visuals**: small, readable plots sized for demos/portfolios
- **Lightweight**: CPU-friendly; no sklearn dependency

## 🚀 Quickstart

```bash
# 1) Create & activate a virtual env (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt
# or minimal:
# pip install streamlit torch transformers nltk rapidfuzz matplotlib pandas numpy

# 3) Run the app
streamlit run app.py
