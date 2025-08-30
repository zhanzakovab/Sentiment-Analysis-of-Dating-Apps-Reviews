# 📊 Sentiment Analysis of Dating App Reviews  

## 📌 Project Overview  
Built a large-scale NLP pipeline to classify sentiment from **600,000+ dating app reviews**, combining **lexical scoring, semantic embeddings, and machine learning models**.  

The project benchmarked classical ML algorithms against a feed-forward neural network and investigated how different preprocessing and feature engineering strategies affect model performance and generalisation.  

---

## 🎯 Motivation  
- User reviews directly shape app reputation, retention, and revenue.  
- Dating app reviews are **noisy**: they contain sarcasm, slang, emojis, and informal grammar — making sentiment classification especially challenging.  
- This project demonstrates how to engineer features, compare models, and identify trade-offs when applying NLP to messy, real-world data.  

---

## 🗂️ Dataset  
- **Source**: [Kaggle – Dating App Reviews](https://www.kaggle.com/datasets/sidharthkriplani/datingappreviews/data)  
- **Initial size**: ~600,000 reviews with both text and star ratings.  
- **Balanced subset**: 25,000 positive + 25,000 negative reviews (to avoid class imbalance and reduce runtime).  
- **Labels**: Collapsed from 3 classes → 2 classes (positive vs negative). Neutral reviews were merged with negative, improving signal clarity and accuracy.  

---

## ⚙️ Preprocessing & Feature Engineering  
- **Text cleaning**: contraction expansion, lowercasing, punctuation/digit removal.  
- **Negation-aware stopwords**: kept “not”, “no”, “nor” to preserve sentiment signals.  
- **Emoji handling**: converted emojis to text descriptors.  
- **Lemmatization with POS tagging**: ensured correct token normalisation.  
- **Vectorization**: TF-IDF for classical models; Word2Vec embeddings for NN.  
- **Lexical features**: VADER sentiment scores added as auxiliary input.  
- **N-grams**: unigrams + bigrams captured key sentiment phrases (e.g., *“waste time”*, *“fake profile”*, *“great app”*).  

---

## 🤖 Models Implemented  
1. **Logistic Regression (LogReg)** – simple, efficient baseline.  
2. **Support Vector Machine (SVM)** – robust for high-dimensional text data.  
3. **Bernoulli Naive Bayes (NB)** – fast but limited for contextual sentiment.  
4. **Feed-Forward Neural Network (FFNN)** – tested with embeddings + fine-tuning.  

### Neural Network Fine-Tuning  
- Reduced hidden layer size (128 → 32) to limit overfitting.  
- Added dropout (30%) to improve generalisation.  
- Lowered learning rate (0.001 → 0.0001).  

---

## 📊 Results  

| Model                     | Accuracy | Key Insight |
|----------------------------|----------|-------------|
| Logistic Regression        | ~87%     | Strong, balanced baseline. |
| Support Vector Machine     | ~88%     | Best overall performance. |
| Bernoulli Naive Bayes      | ~78%     | Fast but weak on negatives. |
| Feed-Forward NN (baseline) | <87%     | Overfit training data. |
| Feed-Forward NN (tuned)    | ~86%     | Better generalisation, but still below SVM/LogReg. |

**Key Findings**  
- Classical ML models outperformed FFNN on this dataset due to feature richness and scale.  
- Combining **VADER lexical scores + Word2Vec embeddings** improved input representation.  
- Binary framing (positive vs negative) was more effective than three-way classification, reducing label noise.  

---

## 🛠️ Tech Stack  
- **Python**: pandas, NumPy, scikit-learn, NLTK, gensim, TensorFlow/Keras  
- **NLP Tools**: VADER Sentiment Analyzer, Word2Vec  
- **Data Viz**: matplotlib, seaborn  

---

## 🚀 How to Run  
```bash
# Clone the repo
git clone https://github.com/your-username/dating-app-sentiment.git
cd dating-app-sentiment

# Install dependencies
pip install -r requirements.txt

# Run preprocessing and training
python preprocess.py
python train.py
