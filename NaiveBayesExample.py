import pandas as pd
from bs4 import BeautifulSoup
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
from utils import split_data_by_sentiment, calculate_word_counts, classify_tweet_with_scores, calculate_likelihood, calculate_log_prior

#nltk.download('punkt')
#nltk.download('stopwords')

# Assuming 'Tweets.csv' is located in the specified path
data = pd.read_csv("inputs/Tweets.csv")

# Copy the original DataFrame to ensure the original data is not modified
data_clean = data[data['airline_sentiment_confidence'] > 0.65].copy()
test_data = data[data['airline_sentiment_confidence'] <= 0.65].copy()

# Map string labels to textual values
sentiment_mapping = {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'}
data_clean.loc[:, 'sentiment'] = data_clean['airline_sentiment'].map(sentiment_mapping)
test_data.loc[:, 'sentiment'] = test_data['airline_sentiment'].map(sentiment_mapping)

# Clean the text using BeautifulSoup and store it in a new column 'text_clean'
# data_clean['text_clean'] = data_clean['text'].apply(lambda x: BeautifulSoup(open("inputs/Tweets.csv"), "lxml").text)
data_clean['text_clean'] = data_clean['text'].apply(lambda x: BeautifulSoup(x, "lxml").get_text()) # Try get_text?
test_data['text_clean'] = test_data['text'].apply(lambda x: BeautifulSoup(x, "lxml").get_text())

# Select only the necessary columns 'text_clean' and 'sentiment'
data_clean = data_clean[['text_clean', 'sentiment']]
test_data = test_data[['text_clean', 'sentiment']]

# Display the first few rows of the cleaned DataFrame
print(data_clean.head())
print(test_data.head())

# Assuming data_clean is your DataFrame containing 'text_clean' and 'sentiment' columns
positive_data = split_data_by_sentiment(data_clean, 'positive')
negative_data = split_data_by_sentiment(data_clean, 'negative')
neutral_data = split_data_by_sentiment(data_clean, 'neutral')

# Calculate word counts for tweets with positive sentiment
word_count_positive = calculate_word_counts(positive_data)

# Calculate word counts for tweets with negative sentiment
word_count_negative = calculate_word_counts(negative_data)

# Calculate word counts for tweets with neutral sentiment
word_count_neutral = calculate_word_counts(neutral_data)

# Calculate total number of words for each sentiment category
total_words_positive = sum(word_count_positive.values())
total_words_negative = sum(word_count_negative.values())
total_words_neutral = sum(word_count_neutral.values())
print("There are {0} positive statements".format(total_words_positive))
print("There are {0} negative statements".format(total_words_negative))
print("There are {0} neutral statements".format(total_words_neutral))

# Calculate likelihoods for each sentiment category
log_likelihood_positive = calculate_likelihood(word_count_positive, total_words_positive)
log_likelihood_negative = calculate_likelihood(word_count_negative, total_words_negative)
log_likelihood_neutral = calculate_likelihood(word_count_neutral, total_words_neutral)


# Calculate the log prior for tweets with positive sentiment
log_prior_positive = calculate_log_prior('positive', data_clean)

# Calculate the log prior for tweets with negative sentiment
log_prior_negative = calculate_log_prior('negative', data_clean)

# Calculate the log prior for tweets with neutral sentiment
log_prior_neutral = calculate_log_prior('neutral', data_clean)


# Classify test tweets
for i in range(5):
    testtext = test_data.iloc[i, 0]
    testsentiment = test_data.iloc[i, 1]
    testpred = classify_tweet_with_scores(testtext, log_likelihood_positive, log_likelihood_negative,
                                          log_likelihood_neutral, log_prior_positive, log_prior_negative,
                                          log_prior_neutral)
    print("Test Tweet: {0} \n Prediction: {1} \n Actual Sentiment: {2} \n".format(testtext, testpred, testsentiment))
