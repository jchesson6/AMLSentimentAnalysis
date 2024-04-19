import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import learning_curve



def run_svm(data_file): #dataset should be full file
    # DATA LOADING AND CLEANING
    # data = pd.read_csv(data_file)
    # data_copy = data.copy()
    # # if data_file = Tweets somehow or add different options in preprocessing function (preferred)
    # data_clean = data_clean[data_clean['airline_sentiment_confidence'] > 0.65]
    # data_clean['text_clean'] = BeautifulSoup(open(data_file), 'lxml').text
    # data_clean['sentiment'] = data_clean['airline_sentiment'].apply(lambda x: 0 if x == 'negative' else 1)
    confidence_threshold = 0.65 # allow user to change
    data_clean = utils.preproc_data(data_file, confidence_threshold)
    pd.set_option('display.max_colwidth', None)
    print(data_clean.head())
    train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
    
    # MACHINE LEARNING MODEL (for each dataset create a text clean and sentiment field)
    print("Setting up model...")
    X_train = train['text_clean'].values
    X_test = test['text_clean'].values
    y_train = train['sentiment']
    y_test = test['sentiment']

    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=utils.tokenize,
        lowercase=True,
        ngram_range=(1, 1),
        stop_words=stopwords.words("english"))

    # CROSS-VALIDATION AND GRID SEARCH FOR HYPERPARAMETERS
    print("Cross-validation and grid search...")
    np.random.seed(1)
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    pipeline_svm = make_pipeline(vectorizer,
                                 SVC(probability=True, kernel="linear", class_weight="balanced"))
    grid_svm = GridSearchCV(pipeline_svm,
                            param_grid={'svc__C': [0.01, 0.1, 1]},
                            cv=kfolds,
                            scoring="roc_auc",
                            verbose=1,
                            n_jobs=-1)
    grid_svm.fit(X_train, y_train)
    grid_svm.score(X_test, y_test)

    print("Best grid params: {0}".format(grid_svm.best_params_))
    print("Best grid score: {0}".format(grid_svm.best_score_))


    print("Obtaining results...")
    utils.report_results(grid_svm.best_estimator_, X_test, y_test)

    roc_svm = utils.get_roc_curve(grid_svm.best_estimator_, X_test, y_test)
    roc_plot = utils.plot_roc_curve(roc_svm)

    train_sizes, train_scores, test_scores = \
        learning_curve(grid_svm.best_estimator_, X_train, y_train, cv=5, n_jobs=-1,
                       scoring="roc_auc", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)
    learning_plot = utils.plot_learning_curve(X_train, y_train, train_sizes,
                        train_scores, test_scores, ylim=(0.7, 1.01), figsize=(14, 6))
    plt.show()

    # EXAMPLES
    print("EXAMPLES")
    print("TEXT -> PREDICTION")
    print("flying with @united is always a great experience -> {0}".
          format(grid_svm.predict(["flying with @united is always a great experience"])))
    print("flying with @united is always a great experience. If you don't lose your luggage -> {0}".
          format(
        grid_svm.predict(["flying with @united is always a great experience. If you don't lose your luggage"])))
    print("I love @united. Sorry, just kidding! -> {0}".
          format(grid_svm.predict(["I love @united. Sorry, just kidding!"])))
    print("@united very bad experience! -> {0}".
          format(grid_svm.predict(["@united very bad experience!"])))


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














