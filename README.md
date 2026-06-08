# 📰 Fake News Detection using Machine Learning and Deep Learning

## 📌 Project Overview

This project focuses on detecting fake news articles using Machine Learning and Deep Learning techniques. The objective is to automatically classify news articles as **Real** or **Fake** by analyzing textual content and engineered linguistic features.

The project compares the performance of traditional Machine Learning algorithms and Deep Learning models, along with sentiment analysis techniques, to identify the most effective approach for fake news detection.

---

## 🎯 Objectives

- Build an automated fake news detection system.
- Perform extensive text preprocessing and feature engineering.
- Compare Machine Learning and Deep Learning models.
- Evaluate models using standard classification metrics.
- Analyze the impact of sentiment analysis on prediction performance.

---

## 📂 Dataset

The dataset consists of two publicly available news datasets:

### True News Dataset
- Real news articles
- Categories:
  - Politics News
  - World News

### Fake News Dataset
- Fake news articles
- Categories:
  - News
  - Politics
  - Government News
  - Left-News
  - US News
  - Middle-East

### Dataset Statistics

| Dataset | Records |
|----------|----------|
| True News | 21,417 |
| Fake News | 23,481 |
| Total | 44,898 |

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow / Keras
- NLTK
- TextBlob
- WordCloud
- Seaborn
- Matplotlib
- XGBoost
- CatBoost
- Gensim
- SpaCy

---

## 🔍 Project Workflow

### 1. Data Collection
- Load True and Fake news datasets
- Merge datasets
- Assign labels:
  - Fake = 1
  - True = 0

### 2. Data Preprocessing
- Remove duplicates
- Handle missing values
- Text cleaning
- Remove punctuation
- Remove stopwords
- Lemmatization
- Tokenization

### 3. Exploratory Data Analysis (EDA)
- Subject distribution
- Publication date analysis
- Word frequency analysis
- Word clouds
- Dataset balancing analysis

### 4. Feature Engineering
- TF-IDF Vectorization
- Count Vectorization
- Sentiment Features
- Part-of-Speech Features
- Readability Scores
- Mutual Information Feature Selection

### 5. Model Training

#### Machine Learning Models
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

#### Deep Learning Model
- Bidirectional Long Short-Term Memory (BiLSTM)

### 6. Model Evaluation
Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## 🤖 Models Implemented

| Model | Type |
|---------|---------|
| Naive Bayes | Machine Learning |
| SVM | Machine Learning |
| Random Forest | Ensemble Learning |
| XGBoost | Gradient Boosting |
| BiLSTM | Deep Learning |

---

## ❤️ Sentiment Analysis

Sentiment analysis was incorporated to investigate whether emotional characteristics of news articles improve fake news classification performance.

Techniques used:

- VADER Sentiment Analysis
- TextBlob Sentiment Scores

Sentiment features were combined with textual features and evaluated against baseline models.

---

## 📊 Key Features

- End-to-end NLP pipeline
- Text preprocessing and cleaning
- Feature extraction and selection
- Machine Learning model comparison
- Deep Learning implementation
- Sentiment analysis integration
- Visualization and EDA
- Performance benchmarking

---

## 📈 Results

The models were compared using multiple evaluation metrics.

### Major Findings

- Traditional Machine Learning models achieved strong baseline performance.
- Ensemble methods outperformed simple classifiers.
- BiLSTM demonstrated superior capability in capturing contextual information.
- Sentiment features provided additional predictive value when combined with textual features.
- Deep Learning models achieved the best overall performance for fake news classification.

---

## 📁 Project Structure

```
Fake-News-Detection/
│
├── data/
│   ├── True.csv
│   └── Fake.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── ML_Models.ipynb
│   └── BiLSTM_Model.ipynb
│
├── images/
│   ├── wordclouds/
│   ├── confusion_matrices/
│   └── roc_curves/
│
├── models/
│   ├── svm_model.pkl
│   ├── random_forest.pkl
│   └── bilstm_model.h5
│
├── requirements.txt
├── README.md
└── fake_news_detection.py
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python fake_news_detection.py
```

Or launch the Jupyter notebooks:

```bash
jupyter notebook
```

---

## 📚 Research Contribution

This project was completed as part of the MSc Data Science dissertation at the University of Essex.

**Dissertation Title:**  
*Fake News Detection using Machine Learning and Deep Learning Models*

The research investigates the comparative performance of Machine Learning and Deep Learning techniques for fake news detection and explores the impact of sentiment analysis on classification accuracy.

---

## 🔮 Future Improvements

- Transformer-based models (BERT, RoBERTa)
- Real-time fake news detection system
- Explainable AI (XAI)
- Multilingual fake news detection
- Social media propagation analysis
- Deployment using Flask/FastAPI

---

## 👩‍💻 Author

**Meghana Dhongadi Ashoka**

MSc Data Science (Distinction)  
University of Essex

- Python
- Machine Learning
- Deep Learning
- Natural Language Processing
- Data Analytics

---

## ⭐ If you found this project useful, please consider giving it a star!
