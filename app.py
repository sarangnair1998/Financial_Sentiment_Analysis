# Imported necessary libraries and modules
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import httpx
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import nltk
import string
import re

# Imported NLTK modules for text preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Loaded environment variables from the .env file (e.g., the API key)
load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

# Downloaded the required NLTK data sets if they were not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Initialized the FastAPI application with metadata
app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="API for analyzing financial news sentiment using FinBERT.",
    version="1.0.0"
)

# Defined the response model using Pydantic (this model was used to structure API responses)
class SentimentResponse(BaseModel):
    ticker: str = Field(..., example="AAPL")
    final_sentiment: str = Field(..., example="Bullish")
    average_sentiment_score: float = Field(..., example=0.345)
    most_common_sentiment: str = Field(..., example="Bullish")

# Loaded the FinBERT tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Initialized the stopwords set and lemmatizer used for preprocessing text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocessed the input text by:
    - Converting it to lowercase.
    - Removing URLs.
    - Removing punctuation.
    - Tokenizing the text.
    - Removing stopwords.
    - Lemmatizing tokens.
    Returned the processed text as a single string.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Removed URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # Removed punctuation
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)

def classify_sentiment(text):
    """
    Classified the sentiment of the input text using FinBERT.
    - Tokenized the input and converted it into tensors.
    - Obtained model outputs without computing gradients (in inference mode).
    - Applied softmax to obtain a probability distribution.
    - Calculated the sentiment score as half the difference between positive and negative scores.
    - Determined the sentiment label based on predefined thresholds.
    Returned a tuple containing the sentiment label and the sentiment score.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    negative_score = probs[0].item()
    # neutral_score = probs[1].item()
    positive_score = probs[2].item()
    
    # Calculated the sentiment score
    sentiment_score = (positive_score - negative_score)

    # Determined the sentiment label based on the score
    if sentiment_score >= 0.9:
        sentiment_label = "Bullish"
    elif 0.4 <= sentiment_score < 0.9:
        sentiment_label = "Somewhat-Bullish"
    elif -0.4 <= sentiment_score < 0.4:
        sentiment_label = "Neutral"
    elif -0.9 < sentiment_score < -0.4:
        sentiment_label = "Somewhat-Bearish"
    else:
        sentiment_label = "Bearish"

    return sentiment_label, sentiment_score

def aggregate_sentiment(sentiment_scores, sentiment_labels):
    """
    Aggregated multiple sentiment scores and labels by:
    - Computing the average sentiment score.
    - Determining the most common sentiment label using a counter.
    - Assigning a final sentiment label based on the average score.
    Returned a tuple containing the final sentiment label, the average score, and the most common label.
    """
    if not sentiment_scores:
        return "No Sentiment Data", 0.0, "No Data"
    
    avg_score = sum(sentiment_scores) / len(sentiment_scores)
    most_common_label = Counter(sentiment_labels).most_common(1)[0][0]
    
    if avg_score >= 0.9:
        final_sentiment = "Bullish"
    elif 0.4 <= avg_score < 0.9:
        final_sentiment = "Somewhat-Bullish"
    elif -0.4 <= avg_score < 0.4:
        final_sentiment = "Neutral"
    elif -0.9 < avg_score < -0.4:
        final_sentiment = "Somewhat-Bearish"
    else:
        final_sentiment = "Bearish"

    return final_sentiment, avg_score, most_common_label

@app.get("/", tags=["General"])
def home():
    """
    Provided a basic message at the root endpoint.
    """
    return {"message": "Welcome to the Financial Sentiment Analysis API"}

@app.get("/api/v1/sentiment", response_model=SentimentResponse, tags=["Sentiment Analysis"])
async def get_sentiment(ticker: str):
    """
    Processed a request for sentiment analysis for a given ticker.
    - Built the API URL for Alpha Vantage using the provided ticker.
    - Made an asynchronous HTTP request to Alpha Vantage.
    - Checked for API rate limits or error messages in the response.
    - Preprocessed the title and summary for each news article.
    - Classified sentiment using FinBERT.
    - Aggregated the sentiment scores and labels.
    Returned the aggregated sentiment analysis using the defined response model.
    """
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}'
    
    # Made an asynchronous API call to Alpha Vantage
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        response.raise_for_status()
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error fetching news data: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=response.status_code, detail=f"HTTP error: {e}")

    data = response.json()
    
    # Checked for rate limit or error messages from Alpha Vantage
    if "Note" in data:
        raise HTTPException(status_code=429, detail=f"API rate limit reached: {data['Note']}")
    if "Error Message" in data:
        raise HTTPException(status_code=400, detail=f"API error: {data['Error Message']}")
    
    sentiment_scores = []
    sentiment_labels = []

    # Processed the news articles if available
    if 'feed' in data:
        articles = data['feed']
        if not articles:
            raise HTTPException(status_code=404, detail=f"No news articles found for ticker: {ticker}")
        
        # Iterated through each article
        for article in articles:
            raw_title = article.get('title', 'No title available')
            raw_summary = article.get('summary', 'No summary available')
            
            # Preprocessed the title and summary
            clean_title = preprocess_text(raw_title)
            clean_summary = preprocess_text(raw_summary)
            
            # Combined the preprocessed title and summary
            combined_text = f"{clean_title}. {clean_summary}"
            
            # Classified sentiment using FinBERT
            final_label, final_score = classify_sentiment(combined_text)
            sentiment_scores.append(final_score)
            sentiment_labels.append(final_label)

        # Aggregated the sentiment scores and labels to determine the final result
        final_sentiment, avg_score, most_common_label = aggregate_sentiment(sentiment_scores, sentiment_labels)

        return SentimentResponse(
            ticker=ticker,
            final_sentiment=final_sentiment,
            average_sentiment_score=round(avg_score, 3),
            most_common_sentiment=most_common_label
        )
    else:
        # Raised an error if no news data was found from Alpha Vantage
        raise HTTPException(status_code=404, detail="No news data found from Alpha Vantage.")

# Ran the FastAPI application using Uvicorn when this module was executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
