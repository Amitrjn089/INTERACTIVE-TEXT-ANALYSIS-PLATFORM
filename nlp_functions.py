# nlp_functions.py
from typing import Sequence, List, Union, Dict, Any
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from nltk.util import ngrams
from collections import Counter, defaultdict
import plotly.graph_objects as go
import spacy
from transformers import pipeline
import pandas as pd

# ----- Load heavy models ONCE (module scope) -----
# Make sure you have downloaded the spacy model (see installation steps below)
nlp = spacy.load("en_core_web_sm")

# Emotion classifier: return_all_scores=True makes results predictable (list of lists)
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
emotion_classifier = pipeline(
    "text-classification",
    model=EMOTION_MODEL,
    tokenizer=EMOTION_MODEL,
    return_all_scores=True
)

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_classifier = pipeline(
    "text-classification",
    model=SENTIMENT_MODEL,
    tokenizer=SENTIMENT_MODEL,
    return_all_scores=True
)

TONE_CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
TONE_LABELS = [
    "factual", "opinion", "question", "command", "emotion", "personal experience",
    "suggestion", "story", "prediction", "warning", "instruction", "definition",
    "narrative", "news", "argument"
]

# Create summarizer pipeline once (reuse)
SUMMARIZER = pipeline("summarization", model="facebook/bart-large-cnn")


# ---------------- Wordcloud ----------------
def show_wordcloud(text: Union[str, Sequence[str]]):
    """
    text: either a single string or a list/sequence of tokens/strings.
    Returns: matplotlib.figure.Figure (so Streamlit can display with st.pyplot).
    """
    try:
        if isinstance(text, str):
            text_for_wc = text
        elif isinstance(text, (list, tuple)):
            text_for_wc = " ".join(map(str, text))
        else:
            text_for_wc = str(text)

        wc = WordCloud(width=800, height=400, background_color="white").generate(text_for_wc)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout()
        return fig
    except Exception as e:
        raise RuntimeError(f"Error generating word cloud: {e}")


# ---------------- N-gram plotting ----------------
def plot_top_ngrams_bar_chart(tokens: Sequence[str], gram_n: int = 3, top_n: int = 15):
    """
    tokens: sequence of tokens (strings).
    gram_n: integer n for n-grams.
    """
    try:
        gram_n = int(gram_n)
        if gram_n <= 0:
            raise ValueError("gram_n should be a positive integer")

        tokens = list(tokens)  # ensure sequence
        ngram_list = list(ngrams(tokens, gram_n))
        ngram_counts = Counter(ngram_list).most_common(top_n)

        if not ngram_counts:
            st.warning("No n-grams found for the provided token list.")
            return None

        labels: List[str] = [" ".join(tup) for tup, _ in ngram_counts]
        counts: List[int] = [int(cnt) for _, cnt in ngram_counts]

        fig = go.Figure(data=[go.Bar(x=labels, y=counts, text=counts, textposition="outside")])
        fig.update_layout(
            height=550,
            title=f"Top {len(labels)} {gram_n}-grams",
            xaxis_title="N-gram",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig, use_container_width=True)
        return fig
    except Exception as e:
        raise RuntimeError(f"An error occurred in plot_top_ngrams_bar_chart: {e}")


# ---------------- Chunking ----------------
def split_into_chunks_spacy(text: str, max_length: int = 500) -> List[str]:
    """
    Split `text` into sentence-based chunks where each chunk has (approximately)
    <= max_length characters. Uses spaCy sentence boundary detection.
    """
    if not isinstance(text, str):
        text = str(text)

    doc = nlp(text)
    chunks: List[str] = []
    current_chunk = ""

    for sent in doc.sents:
        sentence = sent.text.strip()
        # +1 for a space when concatenating
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


# ---------------- Emotion detection ----------------
def detect_emotions(text: str) -> pd.DataFrame:
    """
    Returns top emotions with averaged scores across chunks as a DataFrame.
    """
    if not text:
        return pd.DataFrame(columns=["Emotion", "Score"])

    chunks = split_into_chunks_spacy(text, max_length=500)
    emotion_totals: Dict[str, float] = defaultdict(float)
    emotion_counts: Dict[str, int] = defaultdict(int)

    for chunk in chunks:
        results = emotion_classifier(chunk)  # usually returns a list (batch) of lists because return_all_scores=True
        # normalize shape: results may be [[{...}, {...}, ...]] or [{...},{...},...]
        per_chunk = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results

        for res in per_chunk:
            label = res.get("label")
            score = float(res.get("score", 0.0))
            if label:
                emotion_totals[label] += score
                emotion_counts[label] += 1

    # compute averages
    emotion_averages = {}
    for label, total in emotion_totals.items():
        count = emotion_counts.get(label, 1)
        emotion_averages[label] = total / count if count else 0.0

    sorted_emotions = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_emotions[:5]
    df = pd.DataFrame(top_5, columns=["Emotion", "Score"])
    return df


# ---------------- Sentiment analysis (averaged across chunks) ----------------
def detect_overall_sentiment_avg(text: str) -> Dict[str, Any]:
    if not text:
        return {"overall_sentiment": None, "average_scores": {}}

    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    chunks = split_into_chunks_spacy(text, max_length=500)
    if not chunks:
        return {"overall_sentiment": None, "average_scores": {}}

    score_total = {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0}
    chunk_count = len(chunks)

    for chunk in chunks:
        results = sentiment_classifier(chunk)  # returns list-of-lists when return_all_scores=True
        per_chunk = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results
        for res in per_chunk:
            label_key = res.get("label")
            mapped = label_map.get(label_key)
            if mapped:
                score_total[mapped] += float(res.get("score", 0.0))

    avg_score = {lbl: (score_total[lbl] / chunk_count) for lbl in score_total}
    overall_sentiment = max(avg_score, key=avg_score.get)
    return {"overall_sentiment": overall_sentiment, "average_scores": avg_score}


# ---------------- Tone classification ----------------
def classify_custom(text: str) -> Dict[str, Any]:
    result = TONE_CLASSIFIER(text, candidate_labels=TONE_LABELS)
    return {
        "text": text,
        "predicted_category": result["labels"][0] if result.get("labels") else None,
        "score": result["scores"][0] if result.get("scores") else None,
        "all_categories": list(zip(result.get("labels", []), result.get("scores", [])))
    }


# ---------------- Summarization ----------------
def summarize_large_text(text: str, chunk_max_chars: int = 500) -> str:
    """
    Chunk the text (by spaCy sentences), summarize each chunk, then optionally summarize combined summary.
    chunk_max_chars: length in characters for each chunk (passed to split_into_chunks_spacy).
    """
    if not text:
        return ""

    chunks = split_into_chunks_spacy(text, max_length=chunk_max_chars)
    chunk_summaries: List[str] = []

    for chunk in chunks:
        words = len(chunk.split())
        # heuristics for summary length (word-based rough heuristic)
        max_summary_words = min(300, max(30, int(words * 0.7)))
        min_summary_words = min(100, max(20, int(words * 0.3)))
        # call summarizer (it expects token length; this is a rough approximation)
        out = SUMMARIZER(chunk, max_length=max_summary_words, min_length=min_summary_words, do_sample=False)
        chunk_summaries.append(out[0]["summary_text"])

    combined = " ".join(chunk_summaries)
    if not combined:
        return ""

    combined_words = len(combined.split())
    final_max = min(400, max(50, int(combined_words * 0.5)))
    final_min = min(150, max(20, int(combined_words * 0.15)))
    final_out = SUMMARIZER(combined, max_length=final_max, min_length=final_min, do_sample=False)
    return final_out[0]["summary_text"]
