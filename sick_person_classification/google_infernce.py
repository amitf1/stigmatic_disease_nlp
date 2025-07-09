# To run this code you need to install the following dependencies:
# pip install google-genai && pip install pandas
# or use the requirements.txt file
# set the GEMINI_API_KEY environment variable before running the code

import base64
import json
import os
from google import genai
from google.genai import types
import pandas as pd
import numpy as np

def system_setup():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"), ## create api key on google ai studio
    )

    generate_content_config = types.GenerateContentConfig(
    thinking_config = types.ThinkingConfig(
        thinking_budget=0,
    ),
    response_mime_type="application/json",
    response_schema=types.Schema(
        type = types.Type.OBJECT,
        properties = {
            "real_sickness": types.Schema(
                type = types.Type.BOOLEAN,
            ),
            "who_is_sick": types.Schema(
                type = types.Type.STRING,
            ),
            "sentiment": types.Schema(
                type = types.Type.STRING,
            ),
            "emotion": types.Schema(
                type = types.Type.STRING,
            ),
        },
    ),
    system_instruction=[
        types.Part.from_text(text="""For each given tweet, analyse the sentence and the hashtags and decide if the author is speaking on a real sickness or just as a figuer of speach. 
    If it is a real sickness, who is the sick person? Is it the author, friends or colleagues, family, a celebrity, other person or people or unclear.
    What's the sentiment of the tweet? Is it positive, negative or neutral.
    What's the emotion of the tweet? Is it surprise, fear, anger, sadness, joy, disgust, or other.
    For each question answer with only one choice and only the choice itself. Make sure to stick with the given choices."""),
    ],
    )
    return client, generate_content_config

def generate(input_text, client, model, generate_content_config):
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
    ]

    output = []
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        output.append(chunk.text)
    return json.loads("".join(output))

def read_csv(file_path, tweet_column, limit=None, seed=42): 
    df = pd.read_csv(file_path)
    if limit:
        df = df.sample(limit, random_state=seed)
    return df[["tweet_id", tweet_column]]

def inference(csv_path, tweet_column, output_path, limit=None, model="gemini-2.5-flash"):
    results = []
    client, generate_content_config = system_setup()
    tweets = read_csv(csv_path, tweet_column, limit)
    for index, row in tweets.iterrows():
        result = generate(row[tweet_column], client, model, generate_content_config)
        result["tweet_id"] = row["tweet_id"]
        result["tweet"] = row[tweet_column]
        results.append(result)
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    csv_path = r"C:\Users\97254\Documents\all_tweets\new clusters\flu_with_clusters.csv"
    tweet_column = "Tweet_Text"
    model = "gemini-2.5-flash"
    os.makedirs("outputs", exist_ok=True)
    limit = 200
    limit_str =  str(limit) if limit else "all"
    output_path = f"outputs/{os.path.basename(csv_path).split('.')[0]}_{limit_str}_{model}_with_sentiment_and_emotion.csv"
    inference(csv_path, tweet_column, output_path, limit=limit, model=model)


