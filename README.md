# Financial Sentiment Analysis API

This project is a prototype API that analyzes financial news sentiment using FinBERT. It fetches news articles from the Alpha Vantage API, preprocesses the text (using NLTK for tokenization, stopword removal, and lemmatization), classifies the sentiment with FinBERT (via Hugging Face's Transformers), and aggregates the results. The API is built with FastAPI.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation & Setup using Docker](#installation--setup-using-docker)
- [Interactive Documentation](#interactive-documentation)
- [Example API Call](#example-api-call)
- [Project Structure](#project-structure)
- [Next Steps / Future Work](#next-steps--future-work)

---

## Overview

The API performs the following steps:

1. **Fetch News Data:**  
   Calls Alpha Vantage’s `NEWS_SENTIMENT` endpoint using a provided ticker (e.g., AAPL) to retrieve news articles.

2. **Preprocess Text:**  
   Processes each article’s title and summary by:
   - Lowercasing text
   - Removing URLs and punctuation
   - Tokenizing
   - Removing stopwords
   - Lemmatizing tokens

3. **Sentiment Classification:**  
   Uses FinBERT to classify sentiment based on the difference between positive and negative probabilities, assigning labels such as Bullish, Somewhat-Bullish, Neutral, etc.

4. **Aggregation:**  
   Aggregates sentiment scores and labels from multiple articles to compute an average score and determine the most common sentiment.

---

## Prerequisites

- **Docker Desktop:**  
  Ensure Docker is installed on your machine (available for macOS and Windows).  
  [Download Docker Desktop](https://www.docker.com/products/docker-desktop)

- **Alpha Vantage API Key:**  
  Obtain a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

---

## Installation & Setup using Docker

Since the project is containerized, you don't need to install Python dependencies locally.

### 1. Clone the Repository

```bash
git clone https://github.com/robosac333/Financial_Sentiment_Analysis.git
cd financial-sentiment-api
```

### 2. Build the Docker Image

Run the following command in the project directory (which contains the `Dockerfile`):

```bash
docker build -t financial-sentiment-api .
```

### 3. Run the Docker Container

Replace `your_api_key` with your actual Alpha Vantage API key:

```bash
docker run -d -p 8000:8000 -e ALPHA_VANTAGE_API_KEY=your_api_key financial-sentiment-api
```

This will start the API on port 8000.

---

## Interactive Documentation

Once the container is running, access the API documentation in your browser:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Example API Call

The primary endpoint for sentiment analysis is:

```
GET /api/v1/sentiment?ticker=<TICKER>
```

### Example using `curl`:

```bash
curl "http://localhost:8000/api/v1/sentiment?ticker=AAPL"
```

**Expected JSON Response:**

```json
{
  "ticker": "AAPL",
  "final_sentiment": "Bullish",
  "average_sentiment_score": 0.345,
  "most_common_sentiment": "Bullish"
}
```

