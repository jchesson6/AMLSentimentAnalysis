import classify
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_dict = {"Airline Tweets": 1, "IMDB Reviews": 2, "General Tweets": 3}
input_dict = {'inputs/Tweets.csv': 1, 'inputs/IMDBDataset.csv': 2, 'inputs/TweetsLarge.csv': 3}
pd.options.mode.copy_on_write = True
pd.set_option('display.max_columns', None)
np.random.seed(42)

while True:
    print("\nCLASSIFIER MENU")
    print("------------------------------------")
    print("1 - Run SVM classifier")
    print("2 - Run Naive Bayes classifier")
    print("3 - Run both classifiers for comparison")
    print("4 - Exit\n")
    clf_choice = input('Enter the menu option number you wish to choose: ')

    if clf_choice == '4':
        break

    elif clf_choice == '1' or clf_choice == '2' or clf_choice == '3':
        print("\nDATASET MENU")
        print("------------------------------------")
        print("1 - Airline Tweets [Multi-class] (3,342 kB)")
        print("2 - IMDB Reviews [Binary] (64,661 kB)")
        print("3 - Sample of General Tweets [Binary] (233,207 kB)\n")

        in_choice = list(map(int, input(
            'Enter the datasets you wish to use in order separated by space only: ').split()))

        if clf_choice == '1':

            for choice in in_choice:
                dataset_nm = utils.get_key_from_value(dataset_dict, choice)
                data_in = utils.get_key_from_value(input_dict, choice)
                if data_in is None:
                    print("Invalid dataset menu selection {0}. Continuing...".format(choice))
                else:
                    print("\nRunning SVM using dataset {0}...".format(dataset_nm))
                    classify.run_classifier(data_in, dataset_nm, clf_choice)

        elif clf_choice == '2':

            for choice in in_choice:
                dataset_nm = utils.get_key_from_value(dataset_dict, choice)
                data_in = utils.get_key_from_value(input_dict, choice)
                if data_in is None:
                    print("Invalid dataset menu selection {0}. Continuing...".format(choice))
                else:
                    print("\nRunning Naive Bayes using dataset {0}...".format(dataset_nm))
                    classify.run_classifier(data_in, dataset_nm, clf_choice)

        elif clf_choice == '3':

            for choice in in_choice:
                dataset_nm = utils.get_key_from_value(dataset_dict, choice)
                data_in = utils.get_key_from_value(input_dict, choice)
                if data_in is None:
                    print("Invalid dataset menu selection {0}. Continuing...".format(choice))
                else:
                    print("\nRunning SVM using dataset {0}...".format(dataset_nm))
                    classify.run_classifier(data_in, dataset_nm, '1')
                    print("\nRunning Naive Bayes using dataset {0}...".format(dataset_nm))
                    classify.run_classifier(data_in, dataset_nm, '2')

        plt.show(block=False)
    else:
        print("INVALID SELECTION!")
