import requests
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize NLTK VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Step 1: Get stock names or symbols from the user
stock_symbols = []
print("Enter stock names or symbols (e.g., Apple, AAPL). Type 'done' when finished:")

while True:
    query = input("Stock symbol: ").strip()
    if query.lower() == 'done':
        if len(stock_symbols) < 3:
            print("You must enter at least 3 stock symbols. Please continue.")
            continue
        break
    stock_symbols.append(query)

# Step 2: Your NewsAPI key
news_api_key = '7c6e502109f8437eb9ea07e22a5868a4'  # Replace with your actual API key

# Step 3: Query for the news
start_date = '2024-10-01'
end_date = '2024-10-15'

# Initialize a list to store sentiment data for all stocks
all_sentiment_data = []

# Process each stock symbol
for query in stock_symbols:
    # Step 4: Construct the API URL
    url = f'https://newsapi.org/v2/everything?q={query}&from={start_date}&to={end_date}&apiKey={news_api_key}'

    # Step 5: Fetch news articles from the API
    response = requests.get(url)

    # Step 6: Check if the request was successful
    if response.status_code == 200:
        articles = response.json()  # Convert the response to JSON
        articles = articles.get('articles', [])  # Extract articles
    else:
        print(f"Error fetching data for {query}: {response.status_code}")
        continue  # Skip to the next symbol if there's an error

    # Step 7: Create a DataFrame from the articles
    news_df = pd.DataFrame(articles)

    # Step 8: Extract only the relevant columns (title, description, publishedAt)
    concise_df = news_df[['title', 'description', 'publishedAt']].dropna()

    # Step 9: Perform sentiment analysis on the 'title' column
    sentiment_scores = concise_df['title'].apply(lambda title: sid.polarity_scores(title))

    # Convert the list of dictionaries into a DataFrame
    sentiment_df = pd.DataFrame(sentiment_scores.tolist())

    # Step 10: Calculate the average compound score
    average_compound_score = sentiment_df['compound'].mean()

    # Step 11: Calculate total positive, negative, and neutral scores
    total_positive = sentiment_df['pos'].sum()
    total_negative = sentiment_df['neg'].sum()
    total_neutral = sentiment_df['neu'].sum()

    # Step 12: Prepare data in the desired format
    sentiment_data = {
        'stock_symbol': query,
        'compound_score': round(average_compound_score, 4),  # Rounding to 4 decimal places
        'positive_score': round(total_positive, 4),
        'negative_score': round(total_negative, 4),
        'neutral_score': round(total_neutral, 4)
    }
    
    # Append the sentiment data for this stock to the list
    all_sentiment_data.append(sentiment_data)

# Step 13: Create a DataFrame for the output data
output_df = pd.DataFrame(all_sentiment_data)

# Step 14: Save the output DataFrame to a CSV file
output_df.to_csv(f'sentiment_analysis_multiple_stocks.csv', index=False)

# Step 15: Display the results in separate lines
print("\nSentiment Analysis Results:")
for index, row in output_df.iterrows():
    print(f"Stock Symbol: {row['stock_symbol']}")
    print(f"Compound Score: {row['compound_score']}")
    print(f"Positive Score: {row['positive_score']}")
    print(f"Negative Score: {row['negative_score']}")
    print(f"Neutral Score: {row['neutral_score']}")
    print()  # Print a newline for better readability

print(f"Results saved to sentiment_analysis_multiple_stocks.csv")
