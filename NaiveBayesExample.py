import pandas as pd
from bs4 import BeautifulSoup
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math

nltk.download('punkt')
nltk.download('stopwords')

# Assuming 'Tweets.csv' is located in the specified path
data = pd.read_csv("C:/Users/pisan/Downloads/Tweets.csv/Tweets.csv")

# Copy the original DataFrame to ensure the original data is not modified
data_clean = data.copy()

# Filter out rows with confidence less than or equal to 0.65
data_clean = data_clean[data_clean['airline_sentiment_confidence'] > 0.65]

# Map string labels to numerical values
sentiment_mapping = {'negative': 1, 'neutral': 0.5, 'positive': 0}
data_clean['sentiment'] = data_clean['airline_sentiment'].map(sentiment_mapping)

# Clean the text using BeautifulSoup and store it in a new column 'text_clean'
data_clean['text_clean'] = data_clean['text'].apply(lambda x: BeautifulSoup(x, "lxml").text)

# Select only the necessary columns 'text_clean' and 'sentiment'
data_clean = data_clean.loc[:, ['text_clean', 'sentiment']]

# Display the first few rows of the cleaned DataFrame
print(data_clean.head())


def split_data_by_sentiment(data, sentiment):
    """
    Split the data DataFrame into separate lists based on sentiment.

    Parameters:
       data (DataFrame): The input DataFrame containing 'text' and 'sentiment' columns.
       sentiment (int): The sentiment label to filter the data.

    Returns:
        list: A list of text corresponding to the specified sentiment.
    """
    return data[data['sentiment'] == sentiment]['text_clean'].tolist()


# Assuming data_clean is your DataFrame containing 'text_clean' and 'sentiment' columns
positive_data = split_data_by_sentiment(data_clean, 0)
negative_data = split_data_by_sentiment(data_clean, 1)
neutral_data = split_data_by_sentiment(data_clean, 0.5)


def preprocess_tweet(tweet):
    # Convert the tweet to lowercase
    tweet = tweet.lower()

    # Remove punctuation from the tweet
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the tweet into individual words
    tokens = nltk.word_tokenize(tweet)

    # Initialize a Porter stemmer for word stemming
    stemmer = PorterStemmer()

    # Get a set of English stopwords from NLTK
    stopwords_set = set(stopwords.words("english"))

    # Apply stemming to each token and filter out stopwords
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_set]

    # Return the preprocessed tokens
    return tokens


from collections import defaultdict

def calculate_word_counts(tweets):
    # Initialize a defaultdict to store word counts, defaulting to 0 for unseen words
    word_count = defaultdict(int)

    # Iterate through each tweet in the given list of tweets
    for tweet in tweets:
        # Tokenize and preprocess the tweet using the preprocess_tweet function
        tokens = preprocess_tweet(tweet)

        # Iterate through each token in the preprocessed tokens
        for token in tokens:
            # Increment the count for the current token in the word_count dictionary
            word_count[token] += 1

    # Return the word_count dictionary containing word frequencies
    return word_count


# Calculate word counts for tweets with positive sentiment
word_count_positive = calculate_word_counts(positive_data)

# Calculate word counts for tweets with negative sentiment
word_count_negative = calculate_word_counts(negative_data)

# Calculate word counts for tweets with neutral sentiment
word_count_neutral = calculate_word_counts(neutral_data)


def calculate_likelihood(word_count, total_words, laplacian_smoothing=1):
    # Create an empty dictionary to store the likelihood values
    likelihood = {}

    # Get the number of unique words in the vocabulary
    vocabulary_size = len(word_count)

    # Iterate through each word and its corresponding count in the word_count dictionary
    for word, count in word_count.items():
        # Calculate the likelihood using Laplacian smoothing formula
        # Laplacian smoothing is used to handle unseen words in training data
        # The formula is (count + smoothing) / (total_words + smoothing * vocabulary_size)
        likelihood[word] = math.log((count + laplacian_smoothing) / (total_words + laplacian_smoothing * vocabulary_size))

    # Return the calculated likelihood dictionary
    return likelihood


# Calculate total number of words for each sentiment category
total_words_positive = sum(word_count_positive.values())
total_words_negative = sum(word_count_negative.values())
total_words_neutral = sum(word_count_neutral.values())

# Calculate likelihoods for each sentiment category
log_likelihood_positive = calculate_likelihood(word_count_positive, total_words_positive)
log_likelihood_negative = calculate_likelihood(word_count_negative, total_words_negative)
log_likelihood_neutral = calculate_likelihood(word_count_neutral, total_words_neutral)


def calculate_log_prior(sentiment, data):
    # Count the number of tweets with the specified sentiment
    sentiment_count = len(data[data['sentiment'] == sentiment])

    if sentiment_count > 0:
        # Calculate the natural logarithm of the ratio of tweets with the specified sentiment to the total number of tweets
        log_prior = math.log(sentiment_count / len(data))
    else:
        # If there are no tweets with the specified sentiment, assign a very small value to avoid division by zero
        log_prior = -float('inf')

    # Return the calculated log prior
    return log_prior


# Calculate the log prior for tweets with positive sentiment
log_prior_positive = calculate_log_prior(0, data_clean)

# Calculate the log prior for tweets with negative sentiment
log_prior_negative = calculate_log_prior(1, data_clean)

# Calculate the log prior for tweets with neutral sentiment
log_prior_neutral = calculate_log_prior(0.5, data_clean)


def classify_tweet_with_scores(tweet, log_likelihood_positive, log_likelihood_negative, log_likelihood_neutral,
                               log_prior_positive, log_prior_negative, log_prior_neutral):
    # Tokenize and preprocess the input tweet
    tokens = preprocess_tweet(tweet)

    # Calculate the log scores for each sentiment category
    log_score_positive = log_prior_positive + sum([log_likelihood_positive.get(token, -math.inf) for token in tokens])
    log_score_negative = log_prior_negative + sum([log_likelihood_negative.get(token, -math.inf) for token in tokens])
    log_score_neutral = log_prior_neutral + sum([log_likelihood_neutral.get(token, -math.inf) for token in tokens])

    # Store the sentiment scores in a dictionary
    sentiment_scores = {
        'positive': log_score_positive,
        'negative': log_score_negative,
        'neutral': log_score_neutral
    }

    # Determine the predicted sentiment based on the highest sentiment score
    predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)

    # Print the predicted sentiment and sentiment scores
    print("Predicted Sentiment:", predicted_sentiment)
    print("Sentiment Scores:")
    for sentiment, score in sentiment_scores.items():
        print(sentiment, ":", score)

    # Return the predicted sentiment
    return predicted_sentiment
