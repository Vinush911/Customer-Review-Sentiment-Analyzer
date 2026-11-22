# AI-Powered Customer Review Sentiment Analyzer (DistilBERT)

An NLP application that uses a pretrained Hugging Face DistilBERT model fine-tuned on the SST-2 dataset to classify English customer reviews as **positive** or **negative**.

## Features

- **Single Review Analysis:** Analyze individual customer reviews by typing custom text or picking from built-in demo examples. Displays predicted sentiment, model confidence, probability scores, and word count.
- **Batch Analysis:** Upload CSV files with a `review` column to classify multiple reviews simultaneously. Generates summary metrics, a results table, a sentiment distribution bar chart, and downloadable analyzed CSVs.
- **Clean Layout:** Clean, professional two-tab interface ("Single Review" and "Batch Analysis") built with native Streamlit components and standard system typography.
- **Discreet Diagnostics:** Technical pipeline specifications and runtime notes are discreetly accessible in the sidebar by appending `?debug=true` to the URL.

## How to Run

### Windows Quick Start
1. Open the project folder in your terminal.
2. Run `run.bat`, or use the manual instructions below.

### Manual Setup (macOS / Linux / Windows)
1. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate on macOS/Linux:
   source .venv/bin/activate

   # Activate on Windows (Command Prompt):
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open the local address displayed in the terminal (typically `http://localhost:8501`). On the first run, the app will automatically download and cache the DistilBERT model.

## Model Evaluation

Run the evaluation script against the included sample test dataset:
```bash
python evaluate.py
```
This computes evaluation metrics (accuracy, precision, recall, F1 score, confusion matrix, and average latency) and saves the results to `evaluation_results.json`.

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface with Single Review and Batch Analysis tabs |
| `model.py` | DistilBERT model loader and sentiment inference pipeline |
| `evaluate.py` | Evaluation script calculating classification metrics and latency |
| `sample_data/labeled_reviews.csv` | Labeled customer reviews used for evaluation |
| `sample_data/sample_reviews.csv` | Unlabeled customer reviews for batch testing |
| `PROJECT_REPORT.md` | Comprehensive project report and technical documentation |
| `requirements.txt` | Python package dependencies |
| `run.bat` | Windows quick launcher script |

## Model Details

- **Hugging Face Model:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- **Task:** Binary sentiment classification (SST-2)
- **Output:** Positive / Negative with softmax confidence score
- **Max Input Length:** 512 tokens (longer text is automatically truncated)
- **Hardware Acceleration:** Uses CUDA GPU when available, otherwise defaults to CPU

## Notes and Limitations

- **Binary Classification:** The model classifies text strictly as positive or negative. It does not output a neutral class; ambiguous, mixed, or sarcastic feedback will be assigned to whichever polarity has higher probability.
- **Language Scope:** Optimized for English reviews matching the general SST-2 training distribution.
# Customer-Review-Sentiment-Analyzer
