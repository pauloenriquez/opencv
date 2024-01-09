import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    # Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Get the sentiment scores for the input text
    sentiment_scores = sid.polarity_scores(text)

    # Determine the sentiment based on the compound score
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

print("Enter text or list of text (press Enter after each text, type 'done' when finished):")

text_here = []
while True:
    user_input = input()
    if user_input.lower() == 'done':
        break
    text_here.append(user_input)

# Example usage
for text in text_here:
    sentiment_result = analyze_sentiment(text)
    print(f"Sentiment for '{text}': {sentiment_result}")

