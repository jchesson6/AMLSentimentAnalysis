import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.calibration import CalibratedClassifierCV


def run_svm(data_in, dataset_nm):

    # INPUTS
    # data_in - the filepath for the selected dataset
    # dataset_nm - name of the currently selected dataset

    # DATA LOADING AND CLEANING
    data_clean, test_clean = utils.preproc_data(data_in, dataset_nm)
    print("Done Preprocessing")
    # data_clean.loc[data_clean['sentiment'] == 0.65, 'sentiment'] = 1  # for now
    print(data_clean.head(5))
    train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
    
    # MACHINE LEARNING MODEL (for each dataset create a text clean and sentiment field)
    print("\nSetting up model...")
    X_train = train['text_clean'].values
    X_test = test['text_clean'].values
    y_train = train['sentiment']
    y_test = test['sentiment']

    if data_in == 'inputs/IMDBDataset.csv':
        tknzr_func = utils.tokenizeReview
    else:
        tknzr_func = utils.tokenizeTweet

    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=tknzr_func,
        token_pattern=None,
        lowercase=True,
        ngram_range=(1, 1),
        stop_words=stopwords.words("english"))

    # CROSS-VALIDATION AND GRID SEARCH FOR HYPERPARAMETERS
    print("Cross-validation and grid search...")
    svm = LinearSVC(dual="auto", class_weight="balanced", random_state=1, max_iter=10000, multi_class="ovr")
    clf = CalibratedClassifierCV(svm)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    pipeline_svm = make_pipeline(vectorizer, clf)
    # pipeline_svm = make_pipeline(vectorizer,
    #                            SVC(probability=True, kernel="linear", class_weight="balanced"))
    grid_svm = GridSearchCV(pipeline_svm,
                            param_grid={'calibratedclassifiercv__estimator__C': [0.01, 0.1, 1]},
                            cv=kfolds,
                            scoring="roc_auc_ovr",
                            verbose=0,
                            n_jobs=-1, error_score='raise', return_train_score=True)
 
    grid_svm.fit(X_train, y_train)
    grid_svm.score(X_test, y_test)

    print("Best grid parameter: {0}".format(grid_svm.best_params_))
    print("Best grid score: {0}".format(grid_svm.best_score_))
    print("Obtaining results...")
    if dataset_nm == "Airline Tweets":
        utils.report_results_multi(grid_svm.best_estimator_, X_test, y_test)
    else:
        utils.report_results(grid_svm.best_estimator_, X_test, y_test)
    # roc_svm = utils.get_roc_curve_multi(grid_svm.best_estimator_, X_test, y_test)
    # roc_plot = utils.plot_roc_curve(roc_svm, dataset_nm)
    print("Results done")

    title_str = "Learning curve for " + str(dataset_nm)

    # train_sizes, train_scores, test_scores = \
        # learning_curve(grid_svm.best_estimator_, X_train, y_train, cv=5, n_jobs=-1,
                       # scoring="roc_auc", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)
    # learning_plot = utils.plot_learning_curve(X_train, y_train, train_sizes, train_scores, test_scores,
                                              # title=title_str, ylim=(0.7, 1.01), figsize=(14, 6))
    print("Learning Curve done")

    # EXAMPLES
    print("\nEXAMPLES")
    print("------------------------------")
    print("TEXT -> PREDICTION")

    # if dataset_nm == "Airline Tweets":
    #     print("flying with @united is always a great experience -> {0}".
    #           format(grid_svm.predict(["flying with @united is always a great experience"])))
    #     print("flying with @united is always a great experience. If you don't lose your luggage -> {0}".
    #           format(
    #         grid_svm.predict(["flying with @united is always a great experience. If you don't lose your luggage"])))
    #     print("I love @united. Sorry, just kidding! -> {0}".
    #           format(grid_svm.predict(["I love @united. Sorry, just kidding!"])))
    #     print("@united very bad experience! -> {0}".
    #           format(grid_svm.predict(["@united very bad experience!"])))


    num_tests = 5
    test_data = test_clean.sample(n=num_tests, random_state=1)
    for i in range(num_tests):
        test_text = test_data['text_clean'].iloc[i]
        pred = grid_svm.predict([test_text])
        print("{0} -> {1}".format(test_text, pred))


# # DATA LOADING AND CLEANING
# data = pd.read_csv("inputs/Tweets.csv")
# data_clean = data.copy()
# data_clean = data_clean[data_clean['airline_sentiment_confidence'] > 0.65]
# data_clean['sentiment'] = data_clean['airline_sentiment'].apply(lambda x: 1 if x == 'negative' else 0)
# data_clean['text_clean'] = data_clean['text'].apply(lambda x: BeautifulSoup(x, "lxml").text)
# data_clean['text_clean'] = data_clean['text'].apply(foo_bar)
# data_clean['text_clean'] = BeautifulSoup(open(x), 'lxml').text
# data_clean['sentiment'] = data_clean['airline_sentiment'].apply(lambda x: 1 if x == 'negative' else 0)
# data_clean = data_clean.loc[:, ['text_clean', 'sentiment']]
# data_clean.head()


# def tokenize(text):
#     tknzr = TweetTokenizer()
#     return tknzr.tokenize(text)
# 
# 
# def stem(doc):
#     return (SnowballStemmer.stem(w) for w in analyzer(doc))

# en_stopwords = stopwords.words("english")

# CROSS-VALIDATION AND GRID SEARCH FOR HYPERPARAMETERS

# kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# 
# np.random.seed(1)
# 
# pipeline_svm = make_pipeline(vectorizer,
#                              SVC(probability=True, kernel="linear", class_weight="balanced"))

# grid_svm = GridSearchCV(pipeline_svm,
#                         param_grid={'svc__C': [0.01, 0.1, 1]},
#                         cv=kfolds,
#                         scoring="roc_auc",
#                         verbose=1,
#                         n_jobs=-1)
# 
# grid_svm.fit(X_train, y_train)
# grid_svm.score(X_test, y_test)

# print("Best grid params: {0}".format(grid_svm.best_params_))
# print("Best grid score: {0}".format(grid_svm.best_score_))
#
# def report_results(model, X, y):
#     pred_proba = model.predict_proba(X)[:, 1]
#     pred = model.predict(X)
#
#     auc = roc_auc_score(y, pred_proba)
#     acc = accuracy_score(y, pred)
#     f1 = f1_score(y, pred)
#     prec = precision_score(y, pred)
#     rec = recall_score(y, pred)
#     result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
#     return result
# report_results(grid_svm.best_estimator_, X_test, y_test)














