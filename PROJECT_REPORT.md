# MINI PROJECT REPORT
## AI-Powered Customer Review Sentiment Analysis using DistilBERT

**Course:** Natural Language Processing (23AIPC401)  
**Application:** 3 – Customer Review Sentiment Analysis  
**Model:** DistilBERT fine-tuned on SST-2  

---

# 1. PROBLEM STATEMENT AND OBJECTIVES

## 1.1 Problem statement
Online platforms receive large volumes of customer reviews. Manually reading every review to understand customer opinion requires time and effort. This project develops a simple NLP application that uses a pretrained Transformer to classify an English customer review as positive or negative and displays the prediction confidence.

## 1.2 Objectives
1. Understand the use of a pretrained Transformer for sentiment classification.
2. Load a pretrained DistilBERT model and tokenizer using Hugging Face Transformers.
3. Accept a review through a web interface and predict its sentiment.
4. Support batch analysis of reviews from a CSV file.
5. Evaluate the model on a small labeled demonstration dataset.
6. Present predictions and sentiment distribution through a user interface.

## 1.3 Scope
The supplied checkpoint performs binary English sentiment classification. Neutral sentiment detection, multilingual analysis, aspect-based sentiment, and model fine-tuning are outside the current implementation.

# 2. TRANSFORMER MODEL SELECTION AND ARCHITECTURE

## 2.1 Selected model
The project uses `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, a pretrained DistilBERT sequence-classification checkpoint fine-tuned for the Stanford Sentiment Treebank binary task (SST-2). It returns two class scores corresponding to negative and positive sentiment.

## 2.2 Why DistilBERT?
DistilBERT is a distilled, smaller Transformer derived from BERT. It retains the encoder-based contextual language representation approach while reducing model size and inference cost relative to the original BERT-base architecture. This makes it a practical choice for a student demonstration.

## 2.3 Architecture
```text
Review text
    |
    v
Tokenizer (WordPiece)
    |  input token IDs + attention mask
    v
DistilBERT Transformer encoder
(6 Transformer layers; self-attention + feed-forward blocks)
    |
    v
Sequence representation ([CLS] token representation)
    |
    v
Classification head
    |  two logits
    v
Softmax
    |
    +---- Negative probability
    +---- Positive probability
    |
    v
Predicted label + confidence
```

## 2.4 Prediction mathematics
For the input review, the classification head produces logits \(z_0\) and \(z_1\). Softmax converts them into probabilities:

\[P(y=i\mid x)=\frac{e^{z_i}}{\sum_{j=0}^{1}e^{z_j}}\]

The predicted class is:

\[\hat y=\arg\max_{i\in\{0,1\}}P(y=i\mid x)\]

The displayed confidence is the probability assigned to the predicted class. It should not be interpreted as a guarantee of correctness.

# 3. APPLICATION WORKFLOW / SYSTEM ARCHITECTURE

```text
                 User
                  |
        Streamlit Web Interface
           /             \
   Single Review       CSV Batch Upload
           \             /
            Input validation
                  |
           DistilBERT Tokenizer
                  |
          Pretrained DistilBERT
                  |
        Classification + Softmax
                  |
       Positive / Negative output
                  |
       Confidence and visual summary
```

The model is loaded once and cached during the Streamlit session. The application supports one-review prediction and CSV batch prediction.

# 4. IMPLEMENTATION DETAILS AND PYTHON SOURCE CODE

## 4.1 Technologies
- Python for application logic.
- Hugging Face Transformers for pretrained model and tokenizer loading.
- PyTorch for tensor operations and inference.
- Streamlit for the interactive web interface.
- Pandas for CSV handling.
- Matplotlib for sentiment distribution visualization.
- Scikit-learn for evaluation metrics.

## 4.2 Main modules
| Module | Responsibility |
|---|---|
| `model.py` | Loads tokenizer and DistilBERT sequence classifier; predicts sentiment |
| `app.py` | Streamlit UI, input validation, single and batch analysis |
| `evaluate.py` | Runs labeled examples and computes evaluation metrics |

## 4.3 Execution
Install dependencies using `pip install -r requirements.txt`, then launch with `streamlit run app.py`. The first execution downloads the model from Hugging Face and requires internet access.

# 5. SAMPLE INPUTS AND OUTPUTS

| Sample review | Expected interpretation |
|---|---|
| “This product is excellent. I absolutely love it.” | Positive |
| “The item broke after one day. Very disappointing.” | Negative |
| “Customer support was helpful and resolved my issue.” | Positive |
| “The package arrived late and was damaged.” | Negative |

The actual application output includes the predicted label and model confidence. Exact scores depend on the checkpoint and runtime and should be captured from a real execution.

# 6. PERFORMANCE EVALUATION, RESULTS AND OBSERVATIONS

The `evaluate.py` script uses `sample_data/labeled_reviews.csv` and calculates:
- Accuracy: proportion of correctly classified examples.
- Precision: proportion of positive predictions that are correct (reported weighted across classes).
- Recall: proportion of actual examples correctly identified (weighted).
- F1-score: harmonic mean of precision and recall (weighted).
- Confusion matrix: counts of actual versus predicted classes.
- Average inference latency: mean prediction time over the included examples.

Run `python evaluate.py` to generate `evaluation_results.json`. **Do not treat this small hand-labeled demonstration set as a statistically representative benchmark.** Record the metrics produced on your own machine and state the test-set size in your submitted report.

# 7. CONCLUSION, LIMITATIONS AND FUTURE ENHANCEMENTS

## 7.1 Conclusion
The project demonstrates an end-to-end NLP application using a pretrained Transformer. DistilBERT converts review text into contextual representations and a classification head predicts positive or negative sentiment. A Streamlit interface makes the system accessible for individual and batch review analysis.

## 7.2 Limitations
1. **Binary output:** the selected checkpoint has no neutral class; every non-empty review is assigned positive or negative.
2. **Language:** it is an English model and may perform poorly on other languages or code-mixed text.
3. **Sarcasm and context:** subtle sarcasm, mixed opinions, and domain-specific expressions may be misclassified.
4. **Input length:** text is truncated to 512 tokens.
5. **Confidence:** softmax confidence is not necessarily calibrated probability of correctness.
6. **Evaluation sample:** the included small sample is intended for demonstration, not robust scientific conclusions.

## 7.3 Future enhancements
- Add a neutral class using a suitable three-class sentiment model.
- Support multilingual and code-mixed reviews.
- Perform aspect-based sentiment analysis (price, quality, delivery, support).
- Evaluate on a larger public review dataset with a held-out test split.
- Add review filtering, exportable reports, and time-based trend analysis.

# VIVA-VOCE PREPARATION

**Q1. What is sentiment analysis?**  
It is an NLP task that identifies the opinion or emotional polarity expressed in text, such as positive or negative sentiment.

**Q2. Why did you select DistilBERT?**  
It is a compact Transformer derived from BERT. It provides contextual text representations with lower computational requirements than BERT-base, making it suitable for a student-level demonstration.

**Q3. What is a tokenizer?**  
A tokenizer converts raw text into model-compatible tokens and numerical token IDs. It also creates an attention mask to distinguish real tokens from padding.

**Q4. What is self-attention?**  
Self-attention allows each token representation to incorporate information from other tokens in the input sequence, helping the model understand context.

**Q5. What does the classification head do?**  
It maps the Transformer representation to two logits, one for each sentiment class.

**Q6. Why is softmax used?**  
Softmax converts logits into normalized values that sum to one and can be interpreted as the model's class probability scores.

**Q7. Does this application detect neutral sentiment?**  
No. The selected SST-2 checkpoint is binary, so neutral or ambiguous reviews are forced into positive or negative.

**Q8. What is the difference between accuracy and F1-score?**  
Accuracy is the overall fraction of correct predictions. F1 combines precision and recall and is useful when class-wise performance matters.

**Q9. What happens if the review is very long?**  
The tokenizer truncates input beyond the configured maximum of 512 tokens.

**Q10. What are the future improvements?**  
A three-class or multilingual model, aspect-based sentiment, and evaluation on a larger representative dataset are possible improvements.
