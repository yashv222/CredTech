
# feature_engineering/unstructured_features.py
import pandas as pd
from pathlib import Path
from utils.logging_utils import setup_logger
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, USE_LIGHT_NLP, FINBERT_MODEL_NAME

logger = setup_logger("unstructured_features")

def _vader_sentiment(texts):
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        scores = []
        for t in texts:
            s = sia.polarity_scores(str(t)[:512])  # truncate long
            # map compound [-1,1] to [-1,1] (unchanged), keep for weighting
            scores.append(s["compound"])
        return scores
    except Exception as e:
        logger.warning(f"VADER failed: {e}")
        return [0.0] * len(texts)

def _finbert_sentiment(texts):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        tok = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        nlp = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, truncation=True)
        out = nlp(list(map(lambda s: str(s)[:512], texts)))
        # labels are POSITIVE/NEGATIVE/NEUTRAL
        mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0,
                   "POSITIVE": 1.0, "NEUTRAL": 0.0, "NEGATIVE": -1.0}
        return [mapping.get(o["label"], 0.0) * float(o.get("score", 1.0)) for o in out]
    except Exception as e:
        logger.warning(f"FinBERT failed, falling back to VADER: {e}")
        return _vader_sentiment(texts)

def analyze_sentiment() -> Path:
    news_csv = RAW_DATA_DIR / "news.csv"
    if not news_csv.exists():
        logger.warning("news.csv not found; skipping unstructured features.")
        return PROCESSED_DATA_DIR / "daily_sentiment.csv"

    df = pd.read_csv(news_csv)
    if df.empty:
        logger.warning("news.csv empty; skipping.")
        return PROCESSED_DATA_DIR / "daily_sentiment.csv"

    texts = df["title"].fillna("").astype(str).tolist()
    if USE_LIGHT_NLP:
        df["sentiment_score"] = _vader_sentiment(texts)
    else:
        df["sentiment_score"] = _finbert_sentiment(texts)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])

    # aggregate per ticker/day
    daily = (
        df.groupby(["ticker", "date"])["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "avg_sentiment_score"})
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / "news_with_sentiment.csv", index=False, encoding="utf-8")
    daily_path = PROCESSED_DATA_DIR / "daily_sentiment.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8")
    logger.info(f"Saved {daily_path} ({len(daily)} rows)")
    return daily_path

if __name__ == "__main__":
    analyze_sentiment()
