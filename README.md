# Verification-Aware Knowledge Graph Construction (VAKGC)

## 📌 Overview

Verification-Aware Knowledge Graph Construction (VAKGC) is an NLP + LLM framework designed to construct cleaner and more reliable knowledge graphs by combining:

- LLM-based triple extraction
- Hallucination detection
- NLI-based verification
- Relation normalization
- Knowledge graph visualization

The framework focuses on reducing hallucinated or unsupported triples generated during automatic knowledge graph construction.

---

# 🚀 Features

✅ Multi-source dataset collection  
✅ Wikipedia + Web + Gutenberg integration  
✅ LLM-based triple extraction  
✅ Verification using Natural Language Inference (NLI)  
✅ Hallucination filtering  
✅ Baseline vs verified KG comparison  
✅ Graph visualization  
✅ Modular phase-wise pipeline  

---

# 🧠 Framework Pipeline

## Phase 1 — Dataset Setup

- Fetch text from:
  - Wikipedia
  - Web sources
  - Project Gutenberg
- Clean noisy text
- Split into sentences
- Filter low-quality sentences

---

## Phase 2 — Triple Extraction

- Use LLMs (Groq / LLaMA 3.1)
- Extract:
  - Subject
  - Relation
  - Object
- Save structured triples

---

## Phase 3 — Baseline Knowledge Graph

- Store all extracted triples
- Create baseline KG

---

## Phase 4 — Hallucination Detection

- Apply NLI verification
- Check whether triples are supported by source sentences
- Remove unsupported triples

---

## Phase 5 — Relation Normalization

- Normalize inconsistent relation labels
- Improve KG consistency

---

## Phase 6 — Evaluation

Compute:
- Retention Rate
- Hallucination Rate
- Precision Proxy
- Triple reduction statistics

---

## Phase 7 — Knowledge Graph Visualization

Generate:
- Baseline KG
- Verified KG
- Side-by-side graph comparison

---

# 📊 Results

| Metric | Value |
|---|---|
| Baseline Triples | 713 |
| Verified Triples | 413 |
| Removed Hallucinations | 300 |
| Retention Rate | 57.92% |
| Hallucination Rate | 42.08% |
| Precision Proxy | 0.5792 |

---

# 🛠️ Technologies Used

- Python
- spaCy
- Hugging Face Transformers
- NetworkX
- Matplotlib
- Pandas
- Groq API
- LLaMA 3.1

---

# 📂 Project Structure

```bash
Phase 1 → Dataset Setup
Phase 2 → Triple Extraction
Phase 3 → Baseline KG
Phase 4 → Hallucination Detection
Phase 5 → Relation Normalization
Phase 6 → Evaluation
Phase 7 → KG Visualization
```

---

# ▶️ Installation

```bash
pip install wikipedia-api spacy requests beautifulsoup4
pip install transformers torch pandas matplotlib networkx
python -m spacy download en_core_web_sm
```

---

# ▶️ Run The Notebook

Open:

```bash
Verification_Aware_Knowledge_Graph_Construction_(VAKGC)_Framework_1.ipynb
```

Run all phases sequentially.

---

# 📈 Future Improvements

- Entity linking with Wikidata
- Better relation canonicalization
- Confidence scoring
- Human evaluation benchmark
- Graph database integration (Neo4j)
- Batch verification optimization
- Retrieval-Augmented Generation (RAG)

---

# 🎯 Applications

- Knowledge Graph Construction
- Fact Verification
- Hallucination Detection
- Scientific Knowledge Mining
- Educational NLP Systems
- AI Reliability Research

---

# 📜 License

This project is intended for educational and research purposes.

---

# 🤝 Acknowledgements

- Hugging Face Transformers
- spaCy
- Groq API
- Wikipedia API
- Project Gutenberg
- Open-source NLP community
