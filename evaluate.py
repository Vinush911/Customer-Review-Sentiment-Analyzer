"""Evaluate the pretrained model on the small, included labeled demonstration set.
Metrics describe this sample only; they are not a general benchmark of the model.
"""
import json, time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from model import SentimentAnalyzer

DATA="sample_data/labeled_reviews.csv"
OUT="evaluation_results.json"

def main():
    df=pd.read_csv(DATA)
    model=SentimentAnalyzer()
    preds=[]; latencies=[]
    for text in df.review:
        start=time.perf_counter(); result=model.predict(text); latencies.append((time.perf_counter()-start)*1000)
        preds.append(result["sentiment"].lower())
    y_true=df.label.str.lower().tolist()
    result={"project_title":"Customer Review Sentiment Analysis using DistilBERT",
      "model":model.model_name,"test_sample_count":len(df),
      "metrics":{"accuracy":accuracy_score(y_true,preds),
      "precision_weighted":precision_score(y_true,preds,average="weighted",zero_division=0),
      "recall_weighted":recall_score(y_true,preds,average="weighted",zero_division=0),
      "f1_weighted":f1_score(y_true,preds,average="weighted",zero_division=0),
      "average_inference_latency_ms":sum(latencies)/len(latencies)},
      "labels":["negative","positive"],"confusion_matrix":confusion_matrix(y_true,preds,labels=["negative","positive"]).tolist(),
      "classification_report":classification_report(y_true,preds,zero_division=0,output_dict=True),
      "warning":"Small hand-labeled demonstration sample; metrics are illustrative and not a statistically representative benchmark.",
      "predictions":[{"review":r,"actual":a,"predicted":p,"latency_ms":round(t,2)} for r,a,p,t in zip(df.review,y_true,preds,latencies)]}
    with open(OUT,"w",encoding="utf-8") as f: json.dump(result,f,indent=2)
    print(json.dumps(result["metrics"],indent=2)); print(f"Saved {OUT}")
if __name__=="__main__": main()
