import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
import math
from collections import defaultdict
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import learning_curve
import warnings

dataset_dict = {"Airline Tweets": 1, "IMDB Reviews": 2, "Sample of General Customer Service Tweets": 3,
                "General Customer Service Tweets": 4}
input_dict = {'inputs/Tweets.csv': 1, 'inputs/IMDBDataset.csv': 2, 'inputs/sample.csv': 3, 'inputs/twcs.csv': 4}
sentiment_mapping = {'negative': 0, 'neutral': 0.5, 'positive': 1}
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


def train_test_split(df, frac=0.1):
    # get random sample
    test = df.sample(frac=frac, axis=0)

    # get everything but the test sample
    train = df.drop(index=test.index)

    return train, test


def preproc_data(data_in, dataset_nm):
    # Read in data
    data = pd.read_csv(data_in)
    data_clean = data.copy()
    print("Preprocessing {0}...".format(dataset_nm))

    if dataset_nm == "Airline Tweets":
        data_clean['text_clean'] = data_clean['text'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
        data_clean.rename(columns={'airline_sentiment': 'sentiment'}, inplace=True)
        # test_clean = data_clean[data_clean['airline_sentiment_confidence'] <= 0.65]
        data_clean = data_clean[data_clean['airline_sentiment_confidence'] > 0.65]
        # data_clean['sentiment'] = data_clean['airline_sentiment'].apply(lambda x: 0 if x == 'negative' else 1)
        # data_clean.loc[:, 'sentiment'] = data_clean['airline_sentiment'].map(sentiment_mapping)

    elif data_in == 'inputs/IMDBDataset.csv':
        data_clean['text_clean'] = data_clean['review'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

        # data_clean['sentiment'] = data_clean['sentiment'].apply(lambda x: 0 if x == 'negative' else 1)

    elif data_in == 'inputs/sample.csv' or data_in == 'inputs/twcs.csv':  # same datasets
        print("Not ready")
        return None

    else:
        print("No dataset preprocessed")
        return None

    data_clean, test_clean = train_test_split(data_clean)
    return data_clean[['text_clean', 'sentiment']], test_clean[['text_clean', 'sentiment']]


def tokenizeTweet(text):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True,)
    return tknzr.tokenize(text)


def tokenizeReview(text):
    tknzr = RegexpTokenizer(r'[a-zA-Z0-9]+')
    return tknzr.tokenize(text)


def stem(doc):
    return (SnowballStemmer.stem(w) for w in analyzer(doc))


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    print(result)


def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr


def plot_roc_curve(roc, dataset_nm):
    print("Plotting ROC Curve...")
    fpr, tpr = roc
    plt.figure(figsize=(14, 8))
    plt.plot(fpr, tpr, color="red")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for %s' %dataset_nm )
    return plt


def plot_learning_curve(X, y, train_sizes, train_scores, test_scores, title='', ylim=None, figsize=(14,8)):
    print("Plotting Learning Curve...")
    plt.figure(figsize=figsize)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="lower right")
    return plt







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



    



def preprocess_tweet(tweet):
    # Convert the tweet to lowercase
    tweet = tweet.lower()

    # Remove punctuation from the tweet
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the tweet into individual words
    tokens = nltk.word_tokenize(tweet)

    # Initialize a Porter stemmer for word stemming
    stemmer = PorterStemmer() # change to SnowballStemmer or use both and choose the shortest

    # Get a set of English stopwords from NLTK
    stopwords_set = set(stopwords.words("english"))

    # Apply stemming to each token and filter out stopwords
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_set]

    # Return the preprocessed tokens
    return tokens
    
    
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
        likelihood[word] = (count + laplacian_smoothing) / (total_words + laplacian_smoothing * vocabulary_size)

    # Return the calculated likelihood dictionary
    return likelihood


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


def classify_tweet_with_scores(tweet, log_likelihood_positive, log_likelihood_negative, log_likelihood_neutral,
                               log_prior_positive, log_prior_negative, log_prior_neutral, actual_sentiment):
    # Tokenize and preprocess the input tweet
    tokens = preprocess_tweet(tweet)

    # Calculate the log scores for each sentiment category
    log_score_positive = log_prior_positive + sum([log_likelihood_positive.get(token, 0) for token in tokens])
    log_score_negative = log_prior_negative + sum([log_likelihood_negative.get(token, 0) for token in tokens])
    log_score_neutral = log_prior_neutral + sum([log_likelihood_neutral.get(token, 0) for token in tokens])

    # Store the sentiment scores in a dictionary
    sentiment_scores = {
        'positive': log_score_positive,
        'negative': log_score_negative,
        'neutral': log_score_neutral
    }

    # Determine the predicted sentiment based on the highest sentiment score
    predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)

    # Print the predicted sentiment and sentiment scores
    print("\nTweet: {0} \nPrediction: {1} \nActual Sentiment: {2}".format(tweet, predicted_sentiment,
                                                                        actual_sentiment))
    # print("Predicted Sentiment:", predicted_sentiment)
    print("Sentiment Scores:")
    print(sentiment_scores)
    # Return the predicted sentiment
    return predicted_sentiment
