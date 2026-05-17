import os, re, json, ast, math, textwrap, statistics
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from collections import Counter

# Page Configuration
st.set_page_config(
    page_title="VAKGC Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global Color palette 
C = {
    "blue"  : "#4C72B0",
    "green" : "#55A868",
    "red"   : "#C44E52",
    "orange": "#DD8452",
    "purple": "#8172B2",
    "bg"    : "#F8F9FA",
}

# Data Directory
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Data")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(BASE_DIR, "data")

# Data Loaders
def _path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)
 
 
def _file_ok(filename: str) -> bool:
    return os.path.exists(_path(filename))
 
 
@st.cache_data
def load_sentences() -> pd.DataFrame | None:
    if not _file_ok("sentences.csv"):
        return None
    return pd.read_csv(_path("sentences.csv"))
 
 
@st.cache_data
def load_triples() -> pd.DataFrame | None:
    if not _file_ok("triples.csv"):
        return None
    df = pd.read_csv(_path("triples.csv"))
    df["triples_parsed"] = df["triples"].apply(_safe_parse)
    return df
 
 
@st.cache_data
def load_baseline() -> pd.DataFrame | None:
    if not _file_ok("baseline_dataset.csv"):
        return None
    df = pd.read_csv(_path("baseline_dataset.csv"))
    df["triples_parsed"] = df["triples"].apply(_safe_parse)
    return df
 
 
@st.cache_data
def load_baseline_flat() -> pd.DataFrame | None:
    if not _file_ok("baseline_flat.csv"):
        return None
    df = pd.read_csv(_path("baseline_flat.csv"))
    return df.dropna(subset=["subject", "relation", "object"])
 
 
@st.cache_data
def load_verified() -> pd.DataFrame | None:
    if not _file_ok("verified_dataset.csv"):
        return None
    df = pd.read_csv(_path("verified_dataset.csv"))
    df["verified_parsed"] = df["verified_triples"].apply(_safe_parse)
    df["verified_count"]  = df["verified_parsed"].apply(len)
    return df
 
 
@st.cache_data
def load_verified_flat() -> pd.DataFrame | None:
    if not _file_ok("verified_flat.csv"):
        return None
    df = pd.read_csv(_path("verified_flat.csv"))
    return df.dropna(subset=["subject", "relation", "object"])
 
 
@st.cache_data
def load_canonical() -> pd.DataFrame | None:
    if not _file_ok("triples_canonical.csv"):
        return None
    df = pd.read_csv(_path("triples_canonical.csv"))
    df["triples_parsed"] = df["triples"].apply(_safe_parse)
    return df
 
 
@st.cache_data
def load_confidence() -> pd.DataFrame | None:
    if not _file_ok("confidence_report.csv"):
        return None
    return pd.read_csv(_path("confidence_report.csv"))
 
 
def _safe_parse(val):
    """Parse a JSON/repr triple string safely; return [] on failure."""
    if pd.isna(val):
        return []
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
        
# Graph Builder & Renderer
def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in df.iterrows():
        subj = str(row["subject"]).strip()
        rel  = str(row["relation"]).strip()
        obj  = str(row["object"]).strip()
        G.add_node(subj)
        G.add_node(obj)
        if G.has_edge(subj, obj):
            existing = G[subj][obj]["label"]
            if rel not in existing:
                G[subj][obj]["label"] = existing + "\n" + rel
        else:
            G.add_edge(subj, obj, label=rel)
    return G
 
 
def _node_colors(G: nx.DiGraph) -> list:
    subjects = set(u for u, v in G.edges())
    objects  = set(v for u, v in G.edges())
    return [
        C["orange"] if n in subjects and n in objects
        else C["blue"] if n in subjects
        else C["green"]
        for n in G.nodes()
    ]
 
 
def _wrap(text: str, maxlen: int = 18) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= maxlen:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines) or text
 
 
def render_graph(G: nx.DiGraph, title: str, edge_color: str,
                 max_edges: int = 60, figsize=(11, 8), seed: int = 42):
    """Return a Matplotlib Figure of the knowledge graph."""
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=figsize, facecolor=C["bg"])
        ax.set_facecolor(C["bg"])
        ax.text(0.5, 0.5, "No graph data available", ha="center", va="center")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.axis("off")
        return fig

    # Trim if too large
    if G.number_of_edges() > max_edges:
        edges = list(G.edges(data=True))[:max_edges]
        keep_nodes = {u for u, v, _ in edges} | {v for u, v, _ in edges}
        G = G.subgraph(keep_nodes).copy()
        keep_pairs = {(u, v) for u, v, _ in edges}
        for u, v in list(G.edges()):
            if (u, v) not in keep_pairs:
                G.remove_edge(u, v)
 
    pos    = nx.spring_layout(G, seed=seed, k=2.2)
    labels = {n: _wrap(n) for n in G.nodes()}
 
    fig, ax = plt.subplots(figsize=figsize, facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
 
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=_node_colors(G),
                           node_size=900, alpha=0.92)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=6.5, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color,
                           arrows=True, arrowsize=16, arrowstyle="-|>",
                           width=1.3, connectionstyle="arc3,rad=0.08",
                           alpha=0.8, min_source_margin=18, min_target_margin=18)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=nx.get_edge_attributes(G, "label"), ax=ax,
        font_size=5, font_color="#222",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.55, ec="none"),
    )
 
    n, e = G.number_of_nodes(), G.number_of_edges()
    ax.set_title(f"{title}\nNodes: {n}  ·  Edges: {e}",
                 fontsize=10, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    return fig


# Sidebar 
def sidebar():
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Knowledge_Graph.png/320px-Knowledge_Graph.png",
            use_container_width=True,
        )
        st.markdown("## 🔬 VAKGC Framework")
        st.markdown(
            "Verification-Aware Knowledge Graph Construction\n\n"
            "**Authors**\n"
            "- Humaima Anwar (FA25-RAI-011)\n"
            "- Minahil Nadeem (FA25-RAI-014)\n"
            "- Minahil Mehdi (FA25-RAI-015)\n"
        )
        st.divider()
 
        page = st.radio(
            "Navigate",
            [
                "🏠 Overview",
                "📄 Dataset (Phase 1)",
                "⛏️ Triple Extraction (Phase 2)",
                "🗂️ Baseline (Phase 3)",
                "✅ Verification (Phase 4)",
                "📊 Confidence (Phase 5)",
                "🔤 Canonicalization (Phase 6)",
                "📈 Evaluation (Phase 7)",
                "🕸️ Knowledge Graph (Phase 8)",
                "🔍 Search Triples",
            ],
        )
        st.divider()
 
        # Data status indicator
        st.markdown("**📁 Data Status**")
        files = {
            "sentences.csv"        : "Phase 1",
            "triples.csv"          : "Phase 2",
            "baseline_dataset.csv" : "Phase 3",
            "verified_dataset.csv" : "Phase 4",
            "baseline_flat.csv"    : "Phase 3 flat",
            "verified_flat.csv"    : "Phase 4 flat",
            "triples_canonical.csv": "Phase 6",
        }
        for fname, label in files.items():
            icon = "✅" if _file_ok(fname) else "❌"
            st.markdown(f"{icon} `{label}`")
 
    return page


# Page helpers
def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def show_missing(name: str):
    st.warning(f"{name} is missing from the Data folder.")


def searchable_table(df: pd.DataFrame, key: str, columns: list[str] | None = None):
    if df is None:
        return
    query = st.text_input("Search", key=key, placeholder="Type a word or phrase")
    view = df.copy()
    if query:
        mask = view.astype(str).apply(
            lambda col: col.str.contains(query, case=False, na=False, regex=False)
        ).any(axis=1)
        view = view[mask]
    if columns:
        columns = [c for c in columns if c in view.columns]
        view = view[columns]
    st.dataframe(view, use_container_width=True, hide_index=True)


def triples_to_flat(df: pd.DataFrame, triples_col: str = "triples_parsed") -> pd.DataFrame:
    rows = []
    if df is None or triples_col not in df.columns:
        return pd.DataFrame(columns=["sentence_id", "source", "subject", "relation", "object"])

    for _, row in df.iterrows():
        sentence_id = row.get("id", row.get("sentence_id", ""))
        for triple in row.get(triples_col, []):
            if isinstance(triple, dict):
                triple = triple.get("triple", [])
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                rows.append({
                    "sentence_id": sentence_id,
                    "source": row.get("source", ""),
                    "subject": triple[0],
                    "relation": triple[1],
                    "object": triple[2],
                })
    return pd.DataFrame(rows)


def page_overview():
    st.title("VAKGC Framework")
    st.caption("Verification-Aware Knowledge Graph Construction")

    sentences = load_sentences()
    triples = load_triples()
    baseline_flat = load_baseline_flat()
    verified_flat = load_verified_flat()
    canonical = load_canonical()

    metric_row({
        "Sentences": 0 if sentences is None else len(sentences),
        "Extracted triples": 0 if triples is None else int(triples["triple_count"].sum()),
        "Baseline edges": 0 if baseline_flat is None else len(baseline_flat),
        "Verified edges": 0 if verified_flat is None else len(verified_flat),
        "Canonical rows": 0 if canonical is None else len(canonical),
    })

    st.subheader("Pipeline Snapshot")
    status = pd.DataFrame([
        {"Phase": "Dataset", "File": "sentences.csv", "Available": _file_ok("sentences.csv")},
        {"Phase": "Triple extraction", "File": "triples.csv", "Available": _file_ok("triples.csv")},
        {"Phase": "Baseline graph", "File": "baseline_flat.csv", "Available": _file_ok("baseline_flat.csv")},
        {"Phase": "Verification", "File": "verified_flat.csv", "Available": _file_ok("verified_flat.csv")},
        {"Phase": "Canonicalization", "File": "triples_canonical.csv", "Available": _file_ok("triples_canonical.csv")},
    ])
    st.dataframe(status, use_container_width=True, hide_index=True)


def page_dataset():
    st.title("Dataset")
    df = load_sentences()
    if df is None:
        show_missing("sentences.csv")
        return
    metric_row({"Rows": len(df), "Sources": df["source"].nunique(), "Categories": df["category"].nunique()})
    searchable_table(df, "dataset_search")


def page_triple_extraction():
    st.title("Triple Extraction")
    df = load_triples()
    if df is None:
        show_missing("triples.csv")
        return
    metric_row({"Rows": len(df), "Triples": int(df["triple_count"].sum()), "Successful rows": int((df["status"] == "ok").sum())})
    searchable_table(df, "triple_search", ["id", "source", "sentence", "triples", "triple_count", "status"])


def page_baseline():
    st.title("Baseline")
    df = load_baseline()
    flat = load_baseline_flat()
    if df is None or flat is None:
        show_missing("baseline_dataset.csv or baseline_flat.csv")
        return
    metric_row({"Rows": len(df), "Flat triples": len(flat), "Subjects": flat["subject"].nunique(), "Objects": flat["object"].nunique()})
    searchable_table(flat, "baseline_search")


def page_verification():
    st.title("Verification")
    df = load_verified()
    flat = load_verified_flat()
    if df is None or flat is None:
        show_missing("verified_dataset.csv or verified_flat.csv")
        return
    avg_conf = flat["confidence"].mean() if "confidence" in flat.columns else 0
    metric_row({"Rows": len(df), "Verified triples": len(flat), "Average confidence": f"{avg_conf:.2%}"})

    if "confidence" in flat.columns:
        threshold = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        flat = flat[flat["confidence"] >= threshold]
    searchable_table(flat, "verification_search")


def page_confidence():
    st.title("Confidence")
    report = load_confidence()
    verified = load_verified_flat()

    if report is not None:
        searchable_table(report, "confidence_report_search")
        return

    if verified is None or "confidence" not in verified.columns:
        show_missing("confidence_report.csv")
        return

    metric_row({
        "Minimum": f"{verified['confidence'].min():.2%}",
        "Average": f"{verified['confidence'].mean():.2%}",
        "Maximum": f"{verified['confidence'].max():.2%}",
    })
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=C["bg"])
    ax.hist(verified["confidence"], bins=20, color=C["purple"], edgecolor="white")
    ax.set_title("Verified Triple Confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Triple count")
    st.pyplot(fig)


def page_canonicalization():
    st.title("Canonicalization")
    df = load_canonical()
    if df is None:
        show_missing("triples_canonical.csv")
        return
    flat = triples_to_flat(df)
    metric_row({"Rows": len(df), "Canonical triples": len(flat), "Unique relations": flat["relation"].nunique() if not flat.empty else 0})
    searchable_table(df, "canonical_search", ["id", "source", "sentence", "triples", "triple_count", "status"])


def page_evaluation():
    st.title("Evaluation")
    baseline = load_baseline_flat()
    verified = load_verified_flat()
    if baseline is None or verified is None:
        show_missing("baseline_flat.csv or verified_flat.csv")
        return

    baseline_edges = set(zip(baseline["subject"], baseline["relation"], baseline["object"]))
    verified_edges = set(zip(verified["subject"], verified["relation"], verified["object"]))
    retained = len(baseline_edges & verified_edges)
    removed = max(len(baseline_edges - verified_edges), 0)
    retention = retained / len(baseline_edges) if baseline_edges else 0
    metric_row({
        "Baseline unique triples": len(baseline_edges),
        "Verified unique triples": len(verified_edges),
        "Retained": retained,
        "Removed": removed,
        "Retention": f"{retention:.2%}",
    })

    counts = pd.DataFrame({
        "Stage": ["Baseline", "Verified"],
        "Triples": [len(baseline_edges), len(verified_edges)],
    })
    st.bar_chart(counts, x="Stage", y="Triples", color="#4C72B0")


def page_knowledge_graph():
    st.title("Knowledge Graph")
    baseline = load_baseline_flat()
    verified = load_verified_flat()
    graph_choice = st.radio("Graph", ["Verified", "Baseline"], horizontal=True)
    max_edges = st.slider("Maximum edges to draw", 10, 150, 60, 10)

    df = verified if graph_choice == "Verified" else baseline
    if df is None:
        show_missing(f"{graph_choice.lower()}_flat.csv")
        return

    source_options = ["All"] + sorted(df["source"].dropna().astype(str).unique().tolist())
    selected_source = st.selectbox("Source", source_options)
    view = df if selected_source == "All" else df[df["source"].astype(str) == selected_source]

    G = build_graph(view)
    color = C["green"] if graph_choice == "Verified" else C["blue"]
    st.pyplot(render_graph(G, f"{graph_choice} Knowledge Graph", color, max_edges=max_edges))


def page_search():
    st.title("Search Triples")
    baseline = load_baseline_flat()
    verified = load_verified_flat()
    canonical = triples_to_flat(load_canonical())

    frames = []
    for name, df in [("Baseline", baseline), ("Verified", verified), ("Canonical", canonical)]:
        if df is not None and not df.empty:
            part = df.copy()
            part.insert(0, "stage", name)
            frames.append(part)

    if not frames:
        st.warning("No triples are available to search.")
        return

    all_triples = pd.concat(frames, ignore_index=True, sort=False)
    searchable_table(all_triples, "all_triples_search")


def main():
    page = sidebar()
    if "Overview" in page:
        page_overview()
    elif "Dataset" in page:
        page_dataset()
    elif "Triple Extraction" in page:
        page_triple_extraction()
    elif "Baseline" in page:
        page_baseline()
    elif "Verification" in page:
        page_verification()
    elif "Confidence" in page:
        page_confidence()
    elif "Canonicalization" in page:
        page_canonicalization()
    elif "Evaluation" in page:
        page_evaluation()
    elif "Knowledge Graph" in page:
        page_knowledge_graph()
    elif "Search Triples" in page:
        page_search()


if __name__ == "__main__":
    main()
