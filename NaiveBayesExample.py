import pandas as pd
import utils
import math
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from utils import split_data_by_sentiment, calculate_word_counts, classify_tweet_with_scores, calculate_likelihood, calculate_log_prior
sentiment_mapping = {'negative': 0, 'neutral': 0.5, 'positive': 1}


def run_sknb(data_in, dataset_nm):
    data_clean, test_clean = utils.preproc_data(data_in, dataset_nm)
    print("Done Preprocessing")
    print(data_clean.head(3))
    print("\nSetting up model...")

    train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
    X_train = train['text_clean'].values
    X_test = test['text_clean'].values
    y_train = train['sentiment']
    y_test = test['sentiment']

    if data_in == 'inputs/Tweets.csv':
        tknzr_func = utils.tokenizeTweet
    elif data_in == 'inputs/IMDBDataset.csv':
        tknzr_func = utils.tokenizeReview

    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=tknzr_func,
        token_pattern=None,
        lowercase=True,
        ngram_range=(1, 1),
        stop_words=stopwords.words("english"))

    clf = ComplementNB()


    pipeline_nb = make_pipeline(vectorizer, clf)
    pipeline_nb.fit(X_train, y_train)
    pipeline_nb.score(X_test, y_test)
    print("Obtaining results...")
    if dataset_nm == "Airline Tweets":
        utils.report_results_multi(pipeline_nb, X_test, y_test)
    else:
        utils.report_results(pipeline_nb, X_test, y_test)

    num_tests = 5
    test_data = test_clean.sample(n=num_tests, random_state=1)
    for i in range(num_tests):
        test_text = test_data['text_clean'].iloc[i]
        pred = pipeline_nb.predict([test_text])
        print("{0} -> {1}".format(test_text, pred))


def run_nb(data_in, dataset_nm):
    # Assuming 'Tweets.csv' is located in the specified path
    # DATA LOADING AND CLEANING
    data_clean, test_clean = utils.preproc_data(data_in, dataset_nm)
    print("Done Preprocessing")

    # Display the first few rows of the cleaned DataFrames
    #  data_clean.loc[:, 'sentiment'] = data_clean['airline_sentiment'].map(sentiment_mapping)
    print(data_clean.head(3))

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
    likelihood_positive = calculate_likelihood(word_count_positive, total_words_positive)
    likelihood_negative = calculate_likelihood(word_count_negative, total_words_negative)
    likelihood_neutral = calculate_likelihood(word_count_neutral, total_words_neutral)

    # Calculate the log prior for tweets with positive sentiment
    log_prior_positive = calculate_log_prior('positive', data_clean)

    # Calculate the log prior for tweets with negative sentiment
    log_prior_negative = calculate_log_prior('negative', data_clean)

    # Calculate the log prior for tweets with neutral sentiment
    log_prior_neutral = calculate_log_prior('neutral', data_clean)

    # Create a dictionary of log-likelihood values for positive sentiment
    log_likelihood_positive = {word: math.log(prob) for word, prob in likelihood_positive.items()}

    # Create a dictionary of log-likelihood values for negative sentiment
    log_likelihood_negative = {word: math.log(prob) for word, prob in likelihood_negative.items()}

    # Create a dictionary of log-likelihood values for neutral sentiment
    log_likelihood_neutral = {word: math.log(prob) for word, prob in likelihood_neutral.items()}

    # Classify test tweets
    for i in range(20):
        test_text = test_clean.iloc[i, 0]
        test_sentiment = test_clean.iloc[i, 1]
        classify_tweet_with_scores(test_text, log_likelihood_positive, log_likelihood_negative,
                                               log_likelihood_neutral, log_prior_positive, log_prior_negative,
                                               log_prior_neutral, test_sentiment)

