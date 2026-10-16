import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def add_sentiment(input_path="data/synthetic_logs.csv", output_path="data/synthetic_logs.csv"):
    df = pd.read_csv(input_path)
    analyzer = SentimentIntensityAnalyzer()

    df["sentiment_compound"] = df["comm_text"].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])

    RESIGNATION_KEYWORDS = ["quit", "leaving", "resign", "recruiter", "last project", "other job"]
    df["resignation_flag"] = df["comm_text"].str.lower().apply(
        lambda t: int(any(k in t for k in RESIGNATION_KEYWORDS))
    )

    df = df.sort_values(["user_id", "date"])
    df["sentiment_roll_mean"] = (
        df.groupby("user_id")["sentiment_compound"]
        .transform(lambda x: x.shift(1).rolling(window=14, min_periods=5).mean())
    )
    df["sentiment_drop"] = df["sentiment_roll_mean"] - df["sentiment_compound"]

    df.to_csv(output_path, index=False)
    print("Sentiment analysis added.")
    return df

if __name__ == "__main__":
    add_sentiment()