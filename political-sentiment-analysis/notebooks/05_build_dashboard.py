"""
05_build_dashboard.py
-----------------------
Builds a comparative sentiment dashboard: sentiment split by candidate,
polarity distribution, subjectivity vs polarity, and comment volume.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_csv("../data/comments_scored.csv")
df = df.dropna(subset=["cleaned_comments"])

COLORS = {"Bawumia": "#2E5C8A", "Mahama": "#B23A48"}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Public X (Twitter) Post Sentiment — Bawumia vs. Mahama", fontsize=16, fontweight="bold")

# 1. Sentiment split by candidate (%)
sentiment_pct = pd.crosstab(df["candidate"], df["sentiment"], normalize="index") * 100
sentiment_pct = sentiment_pct[["Negative", "Neutral", "Positive"]]
sentiment_pct.plot(kind="bar", stacked=True, ax=axes[0, 0],
                    color=["#B23A48", "#CCCCCC", "#2E5C8A"])
axes[0, 0].set_title("Sentiment Split by Candidate")
axes[0, 0].set_ylabel("% of Posts")
axes[0, 0].set_xlabel("")
axes[0, 0].tick_params(axis="x", rotation=0)
axes[0, 0].legend(title="Sentiment", loc="upper right", bbox_to_anchor=(1.3, 1))
axes[0, 0].yaxis.set_major_formatter(mticker.PercentFormatter())

# 2. Polarity distribution overlay
for candidate in ["Bawumia", "Mahama"]:
    axes[0, 1].hist(df.loc[df["candidate"] == candidate, "polarity"], bins=30,
                     alpha=0.55, label=candidate, color=COLORS[candidate])
axes[0, 1].set_title("Polarity Distribution")
axes[0, 1].set_xlabel("Polarity (−1 = negative, +1 = positive)")
axes[0, 1].set_ylabel("Number of Posts")
axes[0, 1].legend()
axes[0, 1].axvline(0, color="black", linewidth=0.8, linestyle="--")

# 3. Subjectivity vs Polarity scatter
for candidate in ["Bawumia", "Mahama"]:
    sub = df[df["candidate"] == candidate]
    axes[1, 0].scatter(sub["polarity"], sub["subjectivity"], alpha=0.25, s=12,
                        label=candidate, color=COLORS[candidate])
axes[1, 0].set_title("Subjectivity vs. Polarity")
axes[1, 0].set_xlabel("Polarity")
axes[1, 0].set_ylabel("Subjectivity (0 = factual, 1 = opinionated)")
axes[1, 0].legend()

# 4. Comment volume by candidate
volume = df["candidate"].value_counts()
axes[1, 1].bar(volume.index, volume.values, color=[COLORS[c] for c in volume.index])
axes[1, 1].set_title("Post Volume by Candidate\n(sample size — not a popularity measure)")
axes[1, 1].set_ylabel("Number of Posts")
for i, v in enumerate(volume.values):
    axes[1, 1].text(i, v + 50, str(v), ha="center", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("../outputs/dashboard.png", dpi=150, bbox_inches="tight")
print("Saved ../outputs/dashboard.png")
