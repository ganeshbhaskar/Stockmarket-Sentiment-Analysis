# 📘 Stock Market News Sentiment Analysis using LSEG Workspace

This project analyzes real-time financial news data from **LSEG Workspace** to generate sentiment scores, visualize market trends, and study the relationship between news sentiment and stock price movements. It includes automated data extraction, preprocessing, sentiment computation using NLP models, and multiple visualizations (heatmaps, line charts, word clouds) that help understand market behavior.

---

## 🚀 Project Overview

Financial markets react heavily to news.  
This project builds a pipeline that:

1. Fetches **real-time news headlines, summaries, and metadata** from **LSEG Workspace API**.
2. Preprocesses and cleans text data.
3. Computes sentiment using NLP techniques.
4. Fetches stock price history for correlation analysis.
5. Visualizes insights through multiple charts:
   - Sentiment heatmaps  
   - Sentiment trend line charts  
   - Word clouds for positive, negative, and neutral news  
6. Stores processed datasets for further modeling.

This enables analysts, traders, and researchers to quickly understand how sentiment shifts across time and how it affects stock prices.

---

## 📂 Project Structure

project/
│── src/
│ ├── fetch_price.py
│ ├── fetch_news.py
│ ├── preprocess_news.py
│ ├── sentiment_analysis.py
│ ├── build_wordclouds.py
│ ├── visualize_sentiment.py
│ └── utils/
│
│── data/
│ ├── raw/
│ ├── processed/
│ └── outputs/ # charts, word clouds, heatmaps
│
│── README.md
│── .gitignore


---

## 🛠️ Technologies Used

### **Data & APIs**
- LSEG Workspace API

### **Programming**
- Python 3.10+
- Pandas, NumPy
- Matplotlib, Seaborn
- WordCloud
- Scikit-learn (optional)

### **NLP**
- Text cleaning  
- Sentiment scoring (VADER / custom models)

---

## 📡 Data Pipeline

1. **News Fetching** — Using LSEG Workspace SDK  
2. **Price Fetching** — Retrieves OHLC/VWAP using Refinitiv API  
3. **Cleaning & Preprocessing**  
4. **Sentiment Computation**  
5. **Merging With Price Data**  
6. **Visualization**  

---

## 📊 Visualizations Generated

- Sentiment Heatmap  
- Stock Price vs Sentiment Trendline  
- Positive / Negative / Neutral Word Clouds  
- Volume & Price Trends  
- Daily Sentiment Movement Charts  

---

## 🔧 How to Run the Project

### 1. Install dependencies

### 2. Fetch News

### 3. Fetch Stock Prices

### 4. Generate Sentiment

### 5. Produce Visualizations

---

## 📥 Data Inputs and Outputs

### **Input**
- LSEG Workspace news feed  
- Stock price time series  

### **Output**
- Processed datasets  
- Sentiment scores  
- Plots and visualizations  

---

## 🤝 Contributions

Feel free to open issues or submit PRs to improve sentiment scoring or visualizations.

---

## 📜 License
MIT License

EOF

