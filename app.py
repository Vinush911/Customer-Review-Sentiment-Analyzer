"""Streamlit interface for the DistilBERT Customer Review Sentiment Analyzer."""
import os
os.environ.setdefault("USE_TF", "0")
import html
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from model import SentimentAnalyzer, MODEL_NAME

# Configure page settings with a clean, professional title and wide layout
st.set_page_config(page_title="Customer Review Sentiment Analyzer", layout="wide")

# Set to True to always show the diagnostics section in the sidebar.
# When False, it only appears if "?debug=true" is present in the URL.
ALWAYS_SHOW_DIAGNOSTICS = False
SHOW_DIAGNOSTICS = ALWAYS_SHOW_DIAGNOSTICS or str(st.query_params.get("debug", "")).lower() in ("true", "1")

# Clean, subtle styling using standard page spacing
st.markdown("""<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1050px;
}
footer {
    visibility: hidden;
}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading DistilBERT model (first run may take a few minutes)...")
def load_analyzer():
    return SentimentAnalyzer()

st.title("Customer Review Sentiment Analyzer")
st.caption("Pretrained DistilBERT Transformer · Binary sentiment classification")

with st.sidebar:
    st.header("Model Information")
    st.write("**Model:** DistilBERT fine-tuned on SST-2")
    st.code(MODEL_NAME, language="text")
    st.write("**Task:** Positive / Negative classification")
    st.write("**Maximum input:** 512 tokens (longer text is truncated)")
    st.info("The model is binary. It does not have a Neutral class; ambiguous reviews are still assigned Positive or Negative.")
    st.divider()
    st.subheader("Quick demo reviews")
    examples = {
        "Positive – excellent": "The product is excellent. The quality is fantastic and delivery was quick.",
        "Negative – poor quality": "Very disappointing purchase. It stopped working after one day.",
        "Positive – service": "Customer support was helpful and resolved my issue quickly.",
        "Negative – delivery": "The package arrived late and the item was damaged. Terrible experience."
    }
    chosen = st.selectbox("Load an example", ["Choose an example"] + list(examples))

    if SHOW_DIAGNOSTICS:
        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        with st.expander("System Diagnostics", expanded=False):
            d_specs, d_notes = st.tabs(["Pipeline Specs", "Runtime Notes"])
            with d_specs:
                st.subheader("Pipeline specs")
                st.markdown("""```text
Customer review
      ↓
DistilBERT Tokenizer
      ↓
Input IDs + Attention Mask
      ↓
DistilBERT Encoder (6 Transformer layers)
      ↓
[CLS] representation / sequence representation
      ↓
Classification head (2 output logits)
      ↓
Softmax probability distribution
      ↓
Positive or Negative + confidence
```""")
                st.markdown("**Why DistilBERT?** It is a compressed Transformer derived from BERT, designed to retain much of BERT's language understanding while using fewer layers and fewer parameters.")
                st.markdown(r"**Core calculation:** For logits $z_i$, softmax gives $P(y=i|x)=e^{z_i}/\sum_j e^{z_j}$. The class with the largest probability is selected.")
                st.markdown("**Model source:** [Hugging Face model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)")
            with d_notes:
                st.subheader("Runtime notes")
                st.markdown("""**Problem:** Manually reading large numbers of customer reviews is time-consuming. A sentiment classifier can help summarize whether feedback is predominantly positive or negative.

**Objectives:**
- Use a pretrained Transformer for a real NLP task.
- Accept a review and predict its sentiment.
- Support batch review analysis through CSV.
- Evaluate predictions using labeled test examples.
- Present results in an accessible web interface.

**Limitations:** Binary SST-2 classification, English-focused model, possible sarcasm/domain bias, and truncation beyond 512 tokens. The model is not a substitute for human interpretation of nuanced feedback.

**Pipeline walkthrough:** The tokenizer converts text into token IDs. DistilBERT processes contextual relationships through self-attention. The classification head produces two logits; softmax converts them to probabilities, and the higher probability determines the predicted class.""")

try:
    analyzer = load_analyzer()
except Exception as exc:
    st.error("Could not load the pretrained model. Check your internet connection on the first run and install requirements.txt.")
    st.exception(exc)
    st.stop()

st.caption(f"Model ready · Running on {analyzer.device}")

def render_result_card(result, word_count):
    """Renders sentiment prediction results in a clean bordered container."""
    label = str(result["sentiment"]).upper()
    is_pos = label == "POSITIVE"
    color = "#16a34a" if is_pos else "#dc2626"
    conf = float(result["confidence"]) * 100
    pos = float(result["positive_score"]) * 100
    neg = float(result["negative_score"]) * 100

    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(
                f'<div style="font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem;">Predicted Sentiment</div>'
                f'<div style="font-size: 1.5rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;">{html.escape(label.title())}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.metric("Confidence", f"{conf:.2f}%")

        st.progress(float(result["confidence"]))

        c1, c2, c3 = st.columns(3)
        c1.metric("Positive probability", f"{pos:.2f}%")
        c2.metric("Negative probability", f"{neg:.2f}%")
        c3.metric("Input length", f"{word_count} words")

tab1, tab2 = st.tabs(["Single Review", "Batch Analysis"])
with tab1:
    st.subheader("Analyze one customer review")
    default_text = examples.get(chosen, "")
    review = st.text_area("Enter or paste a customer review", value=default_text, height=150,
                          placeholder="Example: The product quality is excellent and delivery was fast...")
    if st.button("Analyze Sentiment", type="primary"):
        try:
            result = analyzer.predict(review)
            render_result_card(result, len(review.split()))
            st.caption("Confidence is the model's softmax score, not a guarantee that the prediction is correct.")
        except ValueError as e:
            st.warning(str(e))

with tab2:
    st.subheader("Analyze multiple reviews from a CSV file")
    st.write("Upload a CSV containing a column named **review**. The app adds predicted sentiment and confidence.")
    uploaded = st.file_uploader("Upload reviews CSV", type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            if "review" not in data.columns:
                st.error("CSV must contain a column named 'review'.")
            else:
                if st.button("Analyze all reviews", type="primary"):
                    with st.spinner("Classifying reviews..."):
                        predictions = [analyzer.predict(str(x)) if pd.notna(x) and str(x).strip() else {"sentiment": "EMPTY", "confidence": 0.0} for x in data.review]
                    data["predicted_sentiment"] = [x["sentiment"] for x in predictions]
                    data["confidence_percent"] = [round(x["confidence"] * 100, 2) for x in predictions]
                    st.dataframe(data, use_container_width=True)
                    counts = data[data.predicted_sentiment != "EMPTY"].predicted_sentiment.value_counts()
                    a, b = st.columns([1, 1])
                    a.metric("Reviews processed", len(data))
                    b.metric("Positive share", f'{(counts.get("POSITIVE", 0) / max(counts.sum(), 1)) * 100:.1f}%')
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    counts.reindex(["POSITIVE", "NEGATIVE"], fill_value=0).plot(
                        kind="bar", ax=ax, color=["#16a34a", "#dc2626"], rot=0
                    )
                    ax.spines[["top", "right"]].set_visible(False)
                    fig.patch.set_alpha(0)
                    ax.set_facecolor("none")
                    ax.set_title("Predicted sentiment distribution")
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Number of reviews")
                    st.pyplot(fig)
                    plt.close(fig)
                    st.download_button("Download analyzed CSV", data.to_csv(index=False).encode("utf-8"),
                                       file_name="sentiment_analysis_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Unable to read or analyze the CSV: {e}")
    st.download_button("Download sample CSV", "review\n\"Absolutely love this product; it works perfectly.\"\n\"Poor quality and terrible customer service.\"\n\"The item arrived on time and matched the description.\"\n", file_name="sample_reviews.csv", mime="text/csv")
