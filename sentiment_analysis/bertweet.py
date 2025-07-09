# Load model directly
# from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import os
from transformers.pipelines import pipeline




# def get_model():
#     tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
#     model = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base")
#     return tokenizer, model




def classify(tweet) #, model, tokenizer):
    # inputs = tokenizer(tweet, return_tensors="pt")
    # outputs = model(**inputs)
    pipe = pipeline("fill-mask", model="vinai/bertweet-base")      
    return pipe(tweet).label

def read_csv(file_path, tweet_column, limit=None, seed=42): 
    df = pd.read_csv(file_path)
    if limit:
        df = df.sample(limit, random_state=seed)
    return df[["tweet_id", tweet_column]]


def inference(csv_path, tweet_column, output_path, limit=None):
    results = []
    # tokenizer, model = get_model()
    tweets = read_csv(csv_path, tweet_column, limit)
    for index, row in tweets.iterrows():
        # result = {"sentiment": classify(row[tweet_column], model, tokenizer)}
        result = {"sentiment": classify(row[tweet_column])}
        result["tweet_id"] = row["tweet_id"]
        result["tweet"] = row[tweet_column]
        results.append(result)
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    csv_path = r"C:\Users\97254\Documents\all_tweets\new clusters\flu_with_clusters.csv"
    tweet_column = "Tweet_Text"
    # model = "gemini-2.5-flash"
    os.makedirs("outputs", exist_ok=True)
    limit = 200
    limit_str =  str(limit) if limit else "all"
    output_path = f"outputs/{os.path.basename(csv_path).split('.')[0]}_sentiment_bertweet.csv"
    inference(csv_path, tweet_column, output_path, limit=None)