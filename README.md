# 10-K Intelligence — RAG assistant for SEC filings

Ask questions about the annual reports (10-Ks) of **Amazon, NVIDIA, Starbucks and JPMorgan** and get
answers grounded in the filings, with source citations.

**Live demo:** _paste your Streamlit link here_

## How it works
PDF ingestion → fixed-size chunking → local `all-MiniLM-L6-v2` embeddings → ChromaDB →
per-company semantic retrieval → Gemini answer grounded only in the retrieved context.

- Per-company retrieval keeps comparison questions balanced across filings.
- Retrieval accuracy is evaluated on a hand-built question set, plus a chunk-size ablation
  (see `01_build_rag.py`, sections 6–7).
- Rate limiting: per-session question cap and a shared throttle to respect the Gemini free tier.

## Run locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # https://10k-intelligence.streamlit.app/
streamlit run app.py
```

## Rebuild the index (optional)
Put the 10-K PDFs in `data/` (named `amazon_10k.pdf`, `nvidia_10k.pdf`, `starbucks_10k.pdf`,
`jpmorgan_10k.pdf`) and run `01_build_rag.py` (needs `pypdf` as well).

**Stack:** Streamlit · ChromaDB · Sentence-Transformers · Google Gemini
