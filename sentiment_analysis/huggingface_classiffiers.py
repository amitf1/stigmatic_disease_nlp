# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
import torch




# def get_model():
#     tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
#     model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base")
#     return tokenizer, model

def get_model_and_tokenizer():
    # Use a pre-trained sentiment analysis model specifically for Twitter
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Standard way to set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode
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
    
    # Map class indices to sentiment labels (this model uses: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive)
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
    csv_path = "/home/amit/nlp/new clusters/flu_with_clusters.csv"
    tweet_column = "Tweet_Text"
    os.makedirs("outputs", exist_ok=True)
    limit = 400
    limit_str =  str(limit) if limit else "all"
    output_path = f"outputs/{os.path.basename(csv_path).split('.')[0]}_sentiment_bertweet_{limit_str}.csv"
    inference(csv_path, tweet_column, output_path, limit=None)