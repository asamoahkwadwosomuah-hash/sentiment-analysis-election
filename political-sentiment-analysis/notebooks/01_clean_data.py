"""
01_clean_data.py
-----------------
Loads and cleans two sets of public X (Twitter) posts discussing Ghana's
two leading 2024 presidential candidates, and combines them into a single
labeled dataset for comparative sentiment and topic analysis.

Inputs : data/bawumia_comments_raw.csv, data/mahama_comments_raw.csv
Output : data/comments_clean.csv
"""

import re
import pandas as pd
import ftfy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def load_comments(path: str, candidate: str) -> pd.DataFrame:
    """Load a raw single-column comments CSV, tagging rows with the candidate."""
    df = pd.read_csv(path, header=None, names=["comments"], on_bad_lines="skip", engine="python")
    df["candidate"] = candidate
    return df


def clean_text(text: str) -> str:
    """
    Lowercase, strip URLs/mentions/hashtags/HTML artifacts, remove punctuation
    and stopwords, and lemmatize. Mirrors the cleaning approach from the
    original capstone notebook.
    """
    text = str(text)
    text = ftfy.fix_text(text)                            # fix mojibake (e.g. smart quotes)
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                     # mentions
    text = re.sub(r"#\w+", "", text)                      # hashtags
    text = re.sub(r"css-\S+", "", text)                   # stray scraping artifacts
    text = re.sub(r"[^A-Za-z\s]", " ", text)              # non-alphabetic characters
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def main():
    bawumia = load_comments("../data/bawumia_comments_raw.csv", "Bawumia")
    mahama = load_comments("../data/mahama_comments_raw.csv", "Mahama")

    print(f"Bawumia raw comments: {len(bawumia)}")
    print(f"Mahama raw comments:  {len(mahama)}")

    df = pd.concat([bawumia, mahama], ignore_index=True)

    # Drop rows that are empty, whitespace-only, or too short to carry meaning
    # (single-word fragments like "dr." or "The" left over from name redaction).
    df["comments"] = df["comments"].astype(str).str.strip()
    df["comments"] = df["comments"].apply(lambda t: ftfy.fix_text(str(t)))
    df = df[df["comments"].str.len() > 3]

    df["cleaned_comments"] = df["comments"].apply(clean_text)

    # Drop rows with no usable text after cleaning
    before = len(df)
    df = df[df["cleaned_comments"].str.len() > 0]
    print(f"Dropped {before - len(df)} rows with no usable text after cleaning")

    df = df.drop_duplicates(subset=["cleaned_comments"]).reset_index(drop=True)

    print(f"\nFinal dataset: {len(df)} comments")
    print(df["candidate"].value_counts())

    df.to_csv("../data/comments_clean.csv", index=False)
    print("\nSaved to ../data/comments_clean.csv")


if __name__ == "__main__":
    main()
