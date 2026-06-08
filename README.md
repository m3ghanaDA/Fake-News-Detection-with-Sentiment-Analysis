# Fake News Detection Using Machine Learning & Deep Learning

An end-to-end natural language processing (NLP) and machine learning framework designed to automatically detect and classify fake news articles. This project benchmarks traditional Machine Learning classifiers against a Deep Learning sequential architecture on a large-scale corpus, integrating structural syntax profiling and sentiment analysis to identify deceptive writing signatures.

---

## 📌 Project Overview
Manual fact-checking is unscalable against the velocity of digital misinformation. Fake news is often engineered with specific psychological triggers, emotional exaggeration, and distinct structural anomalies. This framework provides an automated approach to isolate these patterns using a dataset of nearly 45,000 articles.

By evaluating both feature-engineered statistical classifiers and raw sequence-learning deep neural networks, this project establishes a definitive performance hierarchy for automated deception detection.

---

## 🛠️ Tech Stack & Dependencies
* **Core Language:** Python 3.x
* **Deep Learning:** TensorFlow 2.x, Keras
* **Machine Learning:** Scikit-Learn, XGBoost
* **Natural Language Processing:** NLTK, SpaCy, TextBlob
* **Data Science Infrastructure:** Pandas, NumPy, Matplotlib, Seaborn

---

### 1. Data Engineering & Rigorous Preprocessing
The framework ingests the public **ISOT Fake News Dataset** (44,898 total records), containing verified news from `Reuters.com` alongside flagged fabrications from `Politifact.com`.
* **Data Deduplication:** Identified and purged severe duplicate clusters across text and title vectors to eliminate training bias and prevent validation data leakage.
* **Linguistic Stabilization:** Transformed unstructured inputs via lowercasing, alphanumeric regular expression cleaning, and web URL/HTML stripping.
* **Token Reduction:** Stripped high-frequency, low-value syntax utilizing custom NLTK stopwords, applying Porter Stemming to collapse standard vocabularies to their foundational base forms.

### 2. Hybrid Feature Engineering & Selection
To expose subtle variance in writing mechanics, a multi-tiered engineered feature matrix was constructed:
* **Syntactic Structure (POS Tagging):** Implemented programmatic Part-of-Speech tracking with SpaCy to monitor grammatical density. Deceptive text historically demonstrates isolated patterns in the distribution frequency of adverbs, pronouns, proper nouns, and exclamation flags.
* **Sentiment Intensity Metrics:** Extracted continuous polarity (`[-1.0, 1.0]`) and subjectivity (`[0.0, 1.0]`) coefficients via TextBlob to capture the sensationalism index characteristic of yellow journalism.
* **Dimensionality Maximization:** Applied **Mutual Information (MI) Scoring** to evaluate non-linear feature relationships, selecting the top 15 highest-ranked architectural features to bundle alongside dense Word2Vec document embeddings.

### 3. Model Architecture Strategy
The processed matrix was isolated via an 80/20 partition to backtest performance dynamics across distinct mathematical strategies:
* **Traditional & Probabilistic Classifiers:** Baseline profiling using Gaussian/Multinomial Naive Bayes alongside high-dimensional Support Vector Machine (SVM) decision boundaries.
* **Ensemble Tree Optimizations:** Implemented parallelized Random Forests and gradient-boosted decision trees via XGBoost with custom $L_1$/$L_2$ structural regularization parameters to ingest the engineered feature matrix.
* **Deep Learning Sequence Modeling (BiLSTM):** Assembled a deep sequential neural network mapping words to dense 120-dimensional embedding layers. The sequences are ingested by a **Bidirectional Long Short-Term Memory (BiLSTM)** layer, reading tokens simultaneously in forward and backward configurations to preserve structural syntax dependencies before reaching a dense classification head.

---

## 📊 Performance Matrix & Key Benchmark Insights

| Model Architecture | Test Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | 86.09% | 70.19% | 98.70% | 82.04% | 0.85 |
| **Random Forest** | 96.10% | 95.11% | 96.23% | 95.67% | 0.96 |
| **Support Vector Machine (SVM)** | 97.27% | 96.33% | 97.60% | 96.96% | 0.97 |
| **XGBoost (Top ML Model)** | **98.18%** | **97.77%** | **98.20%** | **97.99%** | **0.98** |
| **Bidirectional LSTM (Deep Learning)** | **99.90%** | **99.90%** | **100.00%** | **99.90%** | **1.00** |

### 🔍 Key Analytic Observations
* **The Dominance of Context:** The **BiLSTM model outperformed all standard ML methods (99.90% Accuracy)**. Because it reads textual semantics in both directions, it successfully captured long-range structural dependencies and semantic nuances that standard n-gram token counts or linear lines ignore.
* **The Sentiment Paradox:** Backtesting the XGBoost classifier **solely on isolated emotional sentiment vectors caused accuracy to collapse to 73.97%**. However, when emotional metrics were *combined* with structural POS counts and text distributions, precision metrics stabilized. This mathematically proves that emotional tone is an unsafe diagnostic tool on its own, but serves as an excellent accelerator when contextualized alongside structural text features.

---

## 📈 Scalability & Future Research Direction
* **Transformer Architectures:** Porting sequential structures to Transformer topologies (e.g., BERT, RoBERTa) to execute multi-headed self-attention operations over larger contexts.
* **Multimodal Streaming Analytics:** Scaling the internal pipelines to parse streaming WebSocket social APIs, extending feature ingestion frameworks to cross-reference localized graphic metadata and image assets alongside plain text.

