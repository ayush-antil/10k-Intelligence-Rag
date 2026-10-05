# %% [markdown]
# # RAG Knowledge Assistant for SEC 10-K Filings
#
# A retrieval-augmented question-answering system over annual reports (10-Ks)
# from four companies spanning different industries: Amazon (e-commerce/cloud),
# Nvidia (semiconductors), Starbucks (consumer/retail), and JPMorgan Chase
# (banking). The variety in industry and filing structure is deliberate — it
# gives retrieval something meaningful to distinguish between, rather than
# four documents that all read the same way.
#
# Pipeline: PDF ingestion → fixed-size chunking → local sentence-transformer
# embeddings → Chroma vector store → semantic retrieval → Gemini-generated,
# context-grounded answers. Section 6 evaluates retrieval accuracy against a
# hand-built question set, and Section 7 compares chunk-size configurations
# to quantify how that parameter affects retrieval quality.

# %%
import os
import glob
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
llm = genai.GenerativeModel("gemini-3.5-flash-lite")  # stable, fast, free-tier friendly

DATA_DIR = "data"          # put your downloaded 10-K PDFs here
COLLECTION_NAME = "tenk_filings"
CHUNK_SIZE = 800            # characters per chunk — this is your tunable knob
CHUNK_OVERLAP = 150

# %% [markdown]
# ## 1. Ingest source documents
# Each 10-K is loaded and its full text extracted. Filenames (minus the
# extension) become the "source" label used later for citations and eval —
# so `amazon_10k.pdf` is tracked simply as `amazon_10k` throughout.

# %%
def load_pdf_text(filepath: str) -> str:
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

pdf_paths = glob.glob(f"{DATA_DIR}/*.pdf")
print(f"Found {len(pdf_paths)} PDFs: {[Path(p).name for p in pdf_paths]}")

documents = {}
for p in pdf_paths:
    name = Path(p).stem
    print(f"Extracting text from {name}...")
    documents[name] = load_pdf_text(p)
    print(f"  done ({len(documents[name]):,} characters)")

# %% [markdown]
# ## 2. Chunk text
# Fixed-size character chunking with overlap, rather than sentence- or
# paragraph-aware splitting. This is a deliberate simplification: it's fast,
# has no external dependencies, and — as shown in Section 7 — its main
# tunable parameter (chunk size) has a measurable, testable effect on
# retrieval accuracy, which is the point of this project's evaluation.

# %%
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

all_chunks = []       # list of chunk strings
all_metadatas = []     # which document + chunk index each came from

for doc_name, text in documents.items():
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    for i, c in enumerate(chunks):
        all_chunks.append(c)
        all_metadatas.append({"source": doc_name, "chunk_id": i})

print(f"Total chunks: {len(all_chunks)}")

# %% [markdown]
# ## 3. Embed and store in Chroma
# Embeddings run locally via `sentence-transformers` (all-MiniLM-L6-v2) —
# fast, free, and avoids sending filing text to a third-party embeddings API.
# Chroma persists to disk (`chroma_db/`), so subsequent runs skip re-embedding
# if the collection already has data — the `collection.count() == 0` check
# below is what makes iterating on later sections fast rather than re-running
# a multi-minute embedding step every time.

# %%
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast, local, good enough

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Only embed if collection is empty (avoid re-embedding every run)
if collection.count() == 0:
    embeddings = embedder.encode(all_chunks, show_progress_bar=True, batch_size=32).tolist()
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    # Chroma has a max batch size per add() call — add in safe-sized batches
    CHROMA_ADD_BATCH = 4000
    for start in range(0, len(all_chunks), CHROMA_ADD_BATCH):
        end = start + CHROMA_ADD_BATCH
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=all_chunks[start:end],
            metadatas=all_metadatas[start:end],
        )
    print(f"Stored {len(all_chunks)} chunks in Chroma")
else:
    print(f"Collection already has {collection.count()} chunks, skipping embed")

# %% [markdown]
# ## 4. Retrieval
# Given a query, emmbed it with the same model used for the chunks and pull
# the top-k nearest chunks by cosine similarity in Chroma. The sanity check
# below confirms retrieval surfaces relevant, correctly-sourced chunks before
# wiring in the LLM — worth verifying independently, since a bad answer could
# otherwise be a retrieval problem, a prompting problem, or a model problem.

# %%
def retrieve(query: str, top_k: int = 4):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return list(zip(results["documents"][0], results["metadatas"][0]))

# quick sanity check
test_results = retrieve("What are the main risk factors mentioned?")
for doc, meta in test_results:
    print(f"[{meta['source']} #{meta['chunk_id']}] {doc[:150]}...\n")

# %% [markdown]
# ## 5. Grounded answer generation
# Retrieved chunks are injected into the prompt as context, with explicit
# instructions to answer only from that context and to decline rather than
# guess when the context doesn't cover the question — a basic but important
# hallucination-mitigation step for a document QA system. The model is also
# asked to cite which source(s) it drew from, which is checked against the
# retrieved sources below.

# %%
def answer_question(query: str, top_k: int = 4) -> dict:
    retrieved = retrieve(query, top_k)
    context = "\n\n---\n\n".join(
        f"[Source: {meta['source']}]\n{doc}" for doc, meta in retrieved
    )

    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say you don't know — do not guess.
Cite which source(s) you used.

Context:
{context}

Question: {query}
"""

    response = llm.generate_content(prompt)
    return {
        "answer": response.text,
        "sources": [meta["source"] for _, meta in retrieved],
    }

result = answer_question("What are the main risk factors mentioned?")
print(result["answer"])
print("Sources:", set(result["sources"]))

# %% [markdown]
# ## 6. Evaluation harness
# Twenty hand-written questions, five per company, each paired with the
# filing that should be retrieved to answer it correctly. This measures
# *retrieval* accuracy specifically (does the right document surface in the
# top-k?) rather than answer quality, which is harder to score automatically
# and is spot-checked manually instead. Retrieval accuracy is the more
# actionable metric here: if the wrong document is retrieved, no amount of
# prompting will produce a correct answer.

# %%
eval_set = [
    # --- Amazon ---
    {"question": "What are Amazon's main risk factors related to competition?",
     "expected_source": "amazon_10k"},
    {"question": "What business segments does Amazon report?",
     "expected_source": "amazon_10k"},
    {"question": "How does Amazon describe risks related to its international operations?",
     "expected_source": "amazon_10k"},
    {"question": "What does Amazon say about risks from its logistics and fulfillment network?",
     "expected_source": "amazon_10k"},
    {"question": "What cybersecurity or data security risks does Amazon disclose?",
     "expected_source": "amazon_10k"},

    # --- Nvidia ---
    {"question": "What risk factors does Nvidia mention about the semiconductor supply chain?",
     "expected_source": "nvidia_10k"},
    {"question": "What does Nvidia say about competition in the AI chip market?",
     "expected_source": "nvidia_10k"},
    {"question": "What export control or geopolitical risks does Nvidia disclose?",
     "expected_source": "nvidia_10k"},
    {"question": "What does Nvidia's filing say about customer concentration risk?",
     "expected_source": "nvidia_10k"},
    {"question": "What intellectual property risks does Nvidia describe?",
     "expected_source": "nvidia_10k"},

    # --- Starbucks ---
    {"question": "What risk factors does Starbucks mention related to commodity prices like coffee?",
     "expected_source": "starbucks_10k"},
    {"question": "What does Starbucks say about labor costs or unionization risk?",
     "expected_source": "starbucks_10k"},
    {"question": "How does Starbucks describe its store expansion strategy?",
     "expected_source": "starbucks_10k"},
    {"question": "What risks does Starbucks mention about consumer spending habits?",
     "expected_source": "starbucks_10k"},
    {"question": "What does Starbucks disclose about supply chain disruptions?",
     "expected_source": "starbucks_10k"},

    # --- JPMorgan ---
    {"question": "What are the main types of risk JPMorgan categorizes in its filing?",
     "expected_source": "jpmorgan_10k"},
    {"question": "What does JPMorgan say about credit risk?",
     "expected_source": "jpmorgan_10k"},
    {"question": "What regulatory or compliance risks does JPMorgan disclose?",
     "expected_source": "jpmorgan_10k"},
    {"question": "How does JPMorgan describe operational risk?",
     "expected_source": "jpmorgan_10k"},
    {"question": "What does JPMorgan say about market risk from interest rate changes?",
     "expected_source": "jpmorgan_10k"},
]

def evaluate(eval_set, top_k=4):
    hits = 0
    for item in eval_set:
        retrieved = retrieve(item["question"], top_k)
        sources_found = {meta["source"] for _, meta in retrieved}
        if item["expected_source"] in sources_found:
            hits += 1
    hit_rate = hits / len(eval_set)
    print(f"Retrieval hit-rate: {hit_rate:.1%} ({hits}/{len(eval_set)})")
    return hit_rate

evaluate(eval_set)

# %% [markdown]
# ### 6b. Harder evaluation set (no company names)
# The eval set above names the company in nearly every question, which makes
# retrieval artificially easy — the company name alone is a strong semantic
# signal almost independent of chunk size. This second set asks the same
# underlying questions without naming the company, closer to how a user
# might actually phrase a question, and is a more genuine test of whether
# chunk size affects retrieval quality.

# %%
eval_set_hard = [
    {"question": "What risks come from being highly dependent on a fulfillment and logistics network?",
     "expected_source": "amazon_10k"},
    {"question": "What risks arise from customer concentration in a hardware supply chain?",
     "expected_source": "nvidia_10k"},
    {"question": "What export control restrictions affect international chip sales?",
     "expected_source": "nvidia_10k"},
    {"question": "How do rising commodity costs like coffee beans affect margins?",
     "expected_source": "starbucks_10k"},
    {"question": "What labor and unionization risks affect a large retail workforce?",
     "expected_source": "starbucks_10k"},
    {"question": "How is credit risk defined for a large financial institution?",
     "expected_source": "jpmorgan_10k"},
    {"question": "What operational risks come from failed internal processes or systems?",
     "expected_source": "jpmorgan_10k"},
    {"question": "What compliance risks come from anti-money laundering regulations?",
     "expected_source": "jpmorgan_10k"},
]

evaluate(eval_set_hard)

# %% [markdown]
# ## 7. Chunk-size ablation
# Chunk size controls a tradeoff: smaller chunks are more precise (less
# irrelevant text per retrieved item) but risk splitting a relevant fact
# across a chunk boundary; larger chunks preserve context but dilute the
# embedding with unrelated text. Rather than guess, this section measures it:
# the same pipeline is rebuilt at several chunk sizes (in a throwaway in-memory
# Chroma, so the persisted `chroma_db/` used by the app is never touched) and
# scored on both evaluation sets above.

# %%
CHUNK_SIZES_TO_TEST = [400, 800, 1600]
OVERLAP_RATIO = CHUNK_OVERLAP / CHUNK_SIZE   # keep overlap proportional to chunk size


def build_temp_collection(chunk_size: int):
    """Chunk + embed every document at `chunk_size` into a throwaway in-memory collection."""
    overlap = int(chunk_size * OVERLAP_RATIO)
    client = chromadb.EphemeralClient()
    name = f"ablation_{chunk_size}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name)

    chunks, metas = [], []
    for doc_name, text in documents.items():
        for i, c in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(c)
            metas.append({"source": doc_name, "chunk_id": i})

    embs = embedder.encode(chunks, batch_size=32, show_progress_bar=True).tolist()
    ids = [f"c{i}" for i in range(len(chunks))]
    for s in range(0, len(chunks), 4000):
        col.add(
            ids=ids[s:s + 4000],
            embeddings=embs[s:s + 4000],
            documents=chunks[s:s + 4000],
            metadatas=metas[s:s + 4000],
        )
    return col, len(chunks)


def hit_rate(col, eval_items, top_k: int = 4) -> float:
    hits = 0
    for item in eval_items:
        q = embedder.encode([item["question"]]).tolist()
        res = col.query(query_embeddings=q, n_results=top_k)
        if item["expected_source"] in {m["source"] for m in res["metadatas"][0]}:
            hits += 1
    return hits / len(eval_items)


# Note: this re-embeds the full corpus once per chunk size, so expect several
# minutes on a CPU-only machine.
ablation_rows = []
for size in CHUNK_SIZES_TO_TEST:
    col, n_chunks = build_temp_collection(size)
    ablation_rows.append(
        {
            "chunk_size": size,
            "n_chunks": n_chunks,
            "easy_hit_rate": hit_rate(col, eval_set),
            "hard_hit_rate": hit_rate(col, eval_set_hard),
        }
    )

print(f"{'chunk size':>10} | {'chunks':>7} | {'easy set':>8} | {'hard set':>8}")
print("-" * 44)
for r in ablation_rows:
    print(
        f"{r['chunk_size']:>10} | {r['n_chunks']:>7,} | "
        f"{r['easy_hit_rate']:>8.1%} | {r['hard_hit_rate']:>8.1%}"
    )

# %% [markdown]
# ### Reading the results
# Compare the *hard* set column first -- the easy set names the company in
# almost every question, so it mostly saturates regardless of chunk size.
# If hit rate rises with chunk size, longer chunks are keeping enough context
# to stay on-topic; if it falls, they are diluting the embedding. Note the
# eval sets are small (20 and 8 questions), so a difference of one question
# is 5% / 12.5% -- treat small gaps as noise rather than a clear winner.
