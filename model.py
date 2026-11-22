"""DistilBERT inference layer for customer-review sentiment classification."""
import os
os.environ.setdefault("USE_TF", "0")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

class SentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("Please enter a customer review.")
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        index = int(torch.argmax(probabilities).item())
        # SST-2 label mapping: LABEL_0 = NEGATIVE, LABEL_1 = POSITIVE.
        label = self.model.config.id2label.get(index, f"LABEL_{index}").upper()
        if label == "LABEL_0": label = "NEGATIVE"
        if label == "LABEL_1": label = "POSITIVE"
        scores = {self.model.config.id2label.get(i, f"LABEL_{i}").upper(): float(p)
                  for i, p in enumerate(probabilities)}
        scores = {("NEGATIVE" if k == "LABEL_0" else "POSITIVE" if k == "LABEL_1" else k): v
                  for k, v in scores.items()}
        return {"sentiment": label, "confidence": float(probabilities[index]),
                "positive_score": float(probabilities[1]),
                "negative_score": float(probabilities[0]), "scores": scores}
