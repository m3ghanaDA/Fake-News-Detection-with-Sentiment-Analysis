# Fake News Detection using Machine Learning and Deep Learning

This repository contains the implementation of my Master's dissertation project at the **University of Essex**, focused on developing an automated framework to detect misinformation with high precision.

## 📌 Project Overview
The proliferation of fake news on social media poses a significant threat to public trust and democracy. This project implements and compares various **Machine Learning** and **Deep Learning** models to identify deceptive news articles by analyzing their linguistic, structural, and emotional characteristics.

Using the **ISOT Fake News Dataset** (~45,000 articles), I developed a pipeline that achieves near-perfect classification accuracy, specifically highlighting the power of deep learning in capturing complex semantic patterns.

## 🚀 Key Features
- **Comprehensive NLP Pipeline:** Advanced text preprocessing including deduplication, lemmatization, and POS tagging.
- **Hybrid Feature Engineering:** Combination of content-based features, structural linguistic markers, and sentiment analysis.
- **Model Benchmarking:** Comparative study of Naive Bayes, SVM, Random Forest, XGBoost, and BiLSTM.
- **Sentiment Integration:** Evaluation of how emotional polarity and subjectivity impact detection efficacy.

## 📊 Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **BiLSTM** | **99.9%** | **99.9%** | **100%** | **99.9%** |
| XGBoost | 98.1% | 97.7% | 98.2% | 97.9% |
| SVM | 97.2% | 96.3% | 97.5% | 96.9% |
| Random Forest| 96.1% | 95.1% | 96.2% | 95.6% |
| Naive Bayes | 86.1% | 70.1% | 98.6% | 82.0% |

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** TensorFlow, Keras, Scikit-learn, NLTK, Spacy, TextBlob
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
- `data/`: Contains links to the ISOT dataset (fake.csv, true.csv).
- `preprocessing/`: Scripts for data cleaning, tokenization, and POS tagging.
- `models/`: Implementations of ML classifiers and the BiLSTM neural network.
- `notebooks/`: Exploratory Data Analysis (EDA) and Model Training logs.

## 🔍 Key Findings
- **Deep Learning Superiority:** The BiLSTM model outperformed ensemble methods by effectively capturing bidirectional contextual dependencies.
- **Sentiment Value:** While sentiment alone provides moderate predictive power (~74% accuracy), it significantly enhances model robustness when combined with structural features.
- **Linguistic Markers:** Fake news consistently shows distinct patterns in part-of-speech distribution (e.g., higher use of pronouns and verbs vs. nouns in real news).

## 🎓 Academic Context
- **Institution:** University of Essex (School of Mathematics, Statistics and Actuarial Science)
- **Course:** MA981 Dissertation
- **Supervisor:** Dr. Tao Gao
