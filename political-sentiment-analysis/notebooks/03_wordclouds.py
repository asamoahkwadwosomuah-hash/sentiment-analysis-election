"""
03_wordclouds.py
------------------
Generates a word cloud of most frequent terms per candidate, plus a
side-by-side comparison figure.
"""

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv("../data/comments_scored.csv")
df = df.dropna(subset=["cleaned_comments"])

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

colors = {"Bawumia": "Blues", "Mahama": "Reds"}

for ax, candidate in zip(axes, ["Bawumia", "Mahama"]):
    text = " ".join(df.loc[df["candidate"] == candidate, "cleaned_comments"].astype(str))
    wc = WordCloud(
        width=900, height=700, background_color="white",
        colormap=colors[candidate], collocations=False, min_font_size=10,
    ).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(f"Most Frequent Terms — {candidate} Comments", fontsize=14, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig("../outputs/wordclouds_comparison.png", dpi=150, bbox_inches="tight")
print("Saved ../outputs/wordclouds_comparison.png")
