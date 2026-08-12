"""
02_sentiment_analysis.py
--------------------------
Runs TextBlob sentiment analysis (polarity + subjectivity) on the cleaned
comments and compares sentiment distribution between the two candidates.
"""

import pandas as pd
from textblob import TextBlob

df = pd.read_csv("../data/comments_clean.csv")
df = df.dropna(subset=["cleaned_comments"]).reset_index(drop=True)


def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity


def get_subjectivity(text):
    return TextBlob(str(text)).sentiment.subjectivity


df["polarity"] = df["cleaned_comments"].apply(get_polarity)
df["subjectivity"] = df["cleaned_comments"].apply(get_subjectivity)


def classify(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"


df["sentiment"] = df["polarity"].apply(classify)

df.to_csv("../data/comments_scored.csv", index=False)

print("=" * 60)
print("Q1. Overall sentiment split (all comments)")
print(df["sentiment"].value_counts())
print((df["sentiment"].value_counts(normalize=True) * 100).round(1))

print("\n" + "=" * 60)
print("Q2. Sentiment split by candidate (%)")
sentiment_by_candidate = pd.crosstab(df["candidate"], df["sentiment"], normalize="index") * 100
print(sentiment_by_candidate.round(1))

print("\n" + "=" * 60)
print("Q3. Average polarity and subjectivity by candidate")
print(df.groupby("candidate")[["polarity", "subjectivity"]].mean().round(3))

print("\n" + "=" * 60)
print("Q4. Sample count per candidate (for context on statistical confidence)")
print(df["candidate"].value_counts())

print("\n" + "=" * 60)
print("Q5. Most positive and most negative comments per candidate (by polarity)")
for cand in df["candidate"].unique():
    sub = df[df["candidate"] == cand]
    print(f"\n--- {cand}: most positive ---")
    print(sub.nlargest(1, "polarity")[["comments", "polarity"]].to_string(index=False))
    print(f"--- {cand}: most negative ---")
    print(sub.nsmallest(1, "polarity")[["comments", "polarity"]].to_string(index=False))
