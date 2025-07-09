# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
import torch

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  
    print(f"Using device: {device}")
    
    return model, tokenizer, device


def classify(tweet, model, tokenizer, device):
    # Use the sentiment analysis model directly
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_map.get(int(predicted_class), "neutral")

def read_csv(file_path, tweet_column, limit=None, seed=42): 
    df = pd.read_csv(file_path)
    if limit:
        df = df.sample(limit, random_state=seed)
    return df[["tweet_id", tweet_column]]


def inference(csv_path, tweet_column, output_path, limit=None):
    results = []
    model, tokenizer, device = get_model_and_tokenizer()
    tweets = read_csv(csv_path, tweet_column, limit)
    for index, row in tweets.iterrows():
        result = {"sentiment": classify(str(row[tweet_column]), model, tokenizer, device)}
        result["tweet_id"] = str(row["tweet_id"])
        result["tweet"] = str(row[tweet_column])
        results.append(result)
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    root_dir = ""
    csv_path = os.path.join(root_dir, "flu_with_clusters.csv")
    tweet_column = "Tweet_Text"
    os.makedirs("outputs", exist_ok=True)
    limit = None #400
    limit_str =  str(limit) if limit else "all"
    output_path = f"outputs/{os.path.basename(csv_path).split('.')[0]}_sentiment_roberta_{limit_str}.csv"
    inference(csv_path, tweet_column, output_path, limit=limit)