"""
04_topic_modeling.py
----------------------
Runs LDA topic modeling separately for each candidate's comments to
surface the main themes discussed, and saves the top words per topic.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv("../data/comments_scored.csv")
df = df.dropna(subset=["cleaned_comments"])

N_TOPICS = 5
N_WORDS = 8


def run_lda(texts, n_topics=N_TOPICS, n_words=N_WORDS):
    vectorizer = CountVectorizer(
        max_df=0.5, min_df=5, token_pattern=r"\b[a-zA-Z]{3,}\b"
    )
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=20, max_iter=15)
    lda.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-n_words:][::-1]]
        topics.append(top_words)
    return topics, dtm.shape


results = {}
for candidate in ["Bawumia", "Mahama"]:
    texts = df.loc[df["candidate"] == candidate, "cleaned_comments"].astype(str)
    texts = texts[texts.str.len() > 0]
    topics, shape = run_lda(texts)
    results[candidate] = topics
    print("=" * 60)
    print(f"{candidate} — document-term matrix shape: {shape}")
    for i, words in enumerate(topics):
        print(f"  Topic {i+1}: {', '.join(words)}")

# Save results to a text file for the README / repo
with open("../outputs/topic_modeling_results.txt", "w") as f:
    for candidate, topics in results.items():
        f.write(f"{candidate} — Top Topics (LDA, {N_TOPICS} topics)\n")
        f.write("=" * 50 + "\n")
        for i, words in enumerate(topics):
            f.write(f"Topic {i+1}: {', '.join(words)}\n")
        f.write("\n")

print("\nSaved ../outputs/topic_modeling_results.txt")
