import numpy as np
import utils
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer

sentiment_mapping = {'negative': 0, 'neutral': 0.5, 'positive': 1}
model_mapping = {'SVM': '1', 'Naive Bayes': '2'}


def run_classifier(data_in, dataset_nm, model):

    # DATA PROCESSING
    data_clean, test_clean = utils.preproc_data(data_in, dataset_nm)
    print("Done Preprocessing")
    print(data_clean.head(3))

    train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
    X_train = train['text_clean'].values
    X_test = test['text_clean'].values
    y_train = train['sentiment']
    y_test = test['sentiment']

    # MODEL SET UP
    print("\nSetting up model...")

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

    if model == '1':  # SVM
        svm = LinearSVC(dual="auto", class_weight="balanced", random_state=1, max_iter=10000, multi_class="ovr")
        clf = CalibratedClassifierCV(svm)
        parameter_grid = {'calibratedclassifiercv__estimator__C': [0.01, 0.1, 1]}
    else:  # NB
        clf = ComplementNB()
        parameter_grid = {'complementnb__alpha': [0.01, 0.1, 1]}

    if dataset_nm == "Airline Tweets":
        scoring = 'roc_auc_ovr'
    else:
        scoring = 'roc_auc'

    print("Cross-validation and grid search...")
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    pipeline_svm = make_pipeline(vectorizer, clf)
    grid_svm = GridSearchCV(pipeline_svm,
                            param_grid=parameter_grid,
                            cv=kfolds,
                            scoring=scoring,
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

    model_name = utils.get_key_from_value(model_mapping, model)
    title_str = "Learning curve for " + str(dataset_nm) + " using " + str(model_name)

    train_sizes, train_scores, test_scores = \
        learning_curve(grid_svm.best_estimator_, X_train, y_train, cv=5, n_jobs=-1, error_score=np.nan,
                       scoring=scoring, train_sizes=np.linspace(.1, 1.0, 10), random_state=1)
    learning_plot = utils.plot_learning_curve(X_train, y_train, train_sizes, train_scores, test_scores,
                                              title=title_str, ylim=(0.7, 1.01), figsize=(14, 6))
    print("Learning Curve done")

    X_final_train = test_clean['text_clean'].values
    y_final_train = test_clean['sentiment']
    test_score = grid_svm.score(X_final_train, y_final_train)

    print("Grid score on unseen test data: {0}".format(test_score))

    print("\nEXAMPLES")
    print("------------------------------")
    print("TEXT -> PREDICTION")

    num_tests = 5
    test_data = test_clean.sample(n=num_tests)
    for i in range(num_tests):
        test_text = test_data['text_clean'].iloc[i]
        pred = grid_svm.predict([test_text])
        print("{0} -> {1}".format(test_text, pred))
