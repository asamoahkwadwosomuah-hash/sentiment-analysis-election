# Public Sentiment Analysis: Ghana 2024 Presidential Candidates (NLP)

**An NLP capstone project — Python, TextBlob, LDA topic modeling — comparing public X (Twitter) post sentiment on Ghana's two leading 2024 presidential candidates.**

## Project background

This was my MSc capstone project (Data Management and Analysis, University of Cape Coast). It analyzes publicly available X (Twitter) posts discussing Ghana's two leading 2024 presidential candidates, Dr. Mahamudu Bawumia and John Dramani Mahama, to compare public sentiment and discussion themes between the two.

**This is a methodology showcase, not a political statement.** The analysis presents what the comment data shows — sentiment scores and topic frequencies — without editorializing on either candidate. Findings reflect the sentiment and language of a specific, non-random sample of X (Twitter) users, not a scientific opinion poll, and should not be read as representative of the Ghanaian electorate as a whole.

**Dataset:** 8,192 cleaned public X (Twitter) posts (6,212 mentioning Bawumia, 1,980 mentioning Mahama), collected via the X API.

## Important limitations (read before drawing conclusions)

- **Sample sizes are unequal** (6,212 vs. 1,980) — this reflects which posts and search terms were sampled, not relative popularity or sentiment volume, and comparisons use percentages rather than raw counts for this reason.
- **X (Twitter) users are not a representative sample** of voters — reply and post sections skew toward more engaged, more opinionated users.
- **TextBlob's sentiment model is a general-purpose lexicon-based tool**, not trained specifically on Ghanaian political discourse, Pidgin English, or local slang — some nuance and sarcasm will be missed.
- **Each candidate's own name was removed from their respective post set** during original data collection (to keep topic modeling focused on substance rather than the search term itself), which occasionally leaves sentence fragments.

## Business/research questions answered

1. What's the overall sentiment split (positive/negative/neutral) in public posts about each candidate?
2. Is there a meaningful difference in average sentiment polarity between the two candidates?
3. Are posts about one candidate more subjective/opinionated than the other?
4. What themes dominate the discussion for each candidate (topic modeling)?
5. What do the most frequent terms associated with each candidate look like?

## Process

1. **Clean** (`notebooks/01_clean_data.py`) — loaded both raw post sets, fixed text encoding issues (mojibake from the original export), removed URLs/mentions/hashtags/scraping artifacts, removed stopwords, lemmatized, and de-duplicated. Final dataset: 8,192 usable posts.
2. **Sentiment score** (`notebooks/02_sentiment_analysis.py`) — used TextBlob to compute polarity (−1 to +1) and subjectivity (0 to 1) for every post, then classified each as Positive / Negative / Neutral.
3. **Word clouds** (`notebooks/03_wordclouds.py`) — generated frequency-based word clouds per candidate to visualize dominant language.
4. **Topic modeling** (`notebooks/04_topic_modeling.py`) — ran Latent Dirichlet Allocation (LDA, 5 topics) separately for each candidate's posts to surface discussion themes.
5. **Dashboard** (`notebooks/05_build_dashboard.py`) — built a 4-panel comparative dashboard covering sentiment split, polarity distribution, subjectivity vs. polarity, and sample size.

## Key findings

- **Sentiment is broadly similar between the two candidates** in this sample: Bawumia posts were 25.9% positive / 17.7% negative / 56.4% neutral; Mahama posts were 22.2% positive / 15.3% negative / 62.5% neutral.
- **Average polarity was nearly identical** (Bawumia: 0.032, Mahama: 0.034) — essentially neutral-to-mildly-positive on average for both, with most of the signal concentrated in a large neutral middle.
- **Mahama posts skewed slightly more neutral** (62.5% vs. 56.4%), while Bawumia posts had a slightly higher share of both positive and negative reactions — suggesting somewhat more polarized engagement.
- **Topic modeling surfaced distinct discussion themes** per candidate: Bawumia's posts clustered around debate performance, the economy, and comparisons to the sitting government; Mahama's posts clustered around policy explanations, the NPP (opposing party), and job creation. Full topic word lists are in `outputs/topic_modeling_results.txt`.
- **Subjectivity was fairly similar** across both candidates, indicating posts for both were a comparable mix of opinion and factual claims.

## Repo structure

```
data/           raw and cleaned post CSVs
notebooks/      cleaning, sentiment, word cloud, topic modeling, and dashboard scripts
outputs/        dashboard.png, wordclouds_comparison.png, topic_modeling_results.txt
```

## Tools used

Python (pandas, TextBlob, NLTK, scikit-learn, gensim-style LDA via scikit-learn, WordCloud, matplotlib), text cleaning and NLP preprocessing, sentiment analysis, unsupervised topic modeling.

---
*This project analyzes publicly posted X (Twitter) content for academic/methodological purposes. It does not represent the political views of the author and should not be interpreted as polling data or a prediction of electoral outcomes.*
