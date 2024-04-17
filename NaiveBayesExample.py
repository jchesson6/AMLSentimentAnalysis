import pandas as pd
from bs4 import BeautifulSoup
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import functions

# nltk.download('punkt')
# nltk.download('stopwords')

# Assuming 'Tweets.csv' is located in the specified path
data = pd.read_csv("inputs/Tweets.csv")

# Copy the original DataFrame to ensure the original data is not modified
data_copy = data.copy()

# Filter out rows with confidence less than or equal to 0.65
data_clean = data_copy[data_copy['airline_sentiment_confidence'] > 0.65]
test_data = data_copy[data_copy['airline_sentiment_confidence'] <= 0.65]

# Map string labels to numerical values
sentiment_mapping = {'negative': 0, 'neutral': 0.5, 'positive': 1}
data_clean['sentiment'] = data_clean['airline_sentiment'].map(sentiment_mapping)
test_data['sentiment'] = test_data['airline_sentiment'].map(sentiment_mapping)

# Clean the text using BeautifulSoup and store it in a new column 'text_clean'
# data_clean['text_clean'] = data_clean['text'].apply(lambda x: BeautifulSoup(open("inputs/Tweets.csv"), "lxml").text)
data_clean['text_clean'] = data_clean['text'].apply(foo_bar)
test_data['text_clean'] = test_data['text'].apply(foo_bar)

# Select only the necessary columns 'text_clean' and 'sentiment'
data_clean = data_clean.loc[:, ['text_clean', 'sentiment']]
test_data = test_data.loc[:, ['text_clean', 'sentiment']]

# Display the first few rows of the cleaned DataFrame
print(data_clean.head())
print(test_data.head())

# Assuming data_clean is your DataFrame containing 'text_clean' and 'sentiment' columns
positive_data = split_data_by_sentiment(data_clean, 1)
negative_data = split_data_by_sentiment(data_clean, 0)
neutral_data = split_data_by_sentiment(data_clean, 0.5)

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
log_prior_positive = calculate_log_prior(1, data_clean)

# Calculate the log prior for tweets with negative sentiment
log_prior_negative = calculate_log_prior(0, data_clean)

# Calculate the log prior for tweets with neutral sentiment
log_prior_neutral = calculate_log_prior(0.5, data_clean)

for i in range(5):
    testtext = test_data.iloc[i, 0]
    testsentiment = test_data.iloc[i, 1]
    testpred = classify_tweet_with_scores(testtext, log_likelihood_positive, log_likelihood_negative,
                                          log_likelihood_neutral, log_prior_positive, log_prior_negative,
                                          log_prior_neutral)
    print("Test Tweet: {0} \n Prediction: {1} \n Actual Sentiment: {2} \n".format(testtext, testpred, testsentiment))
