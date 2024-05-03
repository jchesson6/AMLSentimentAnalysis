import svm
import NaiveBayesExample as NB
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_dict = {"Airline Tweets": 1, "IMDB Reviews": 2, "Sample of General Customer Service Tweets": 3,
                "General Customer Service Tweets": 4}
input_dict = {'inputs/Tweets.csv': 1, 'inputs/IMDBDataset.csv': 2, 'inputs/sample.csv': 3, 'inputs/twcs.csv': 4}
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
        print("3 - Sample of General Customer Service Tweets [Binary] [NOT WORKING] (17 kB)")
        print("4 - General Customer Service Tweets [Binary] [NOT WORKING] (504,403 kB)\n")

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
                    svm.run_svm(data_in, dataset_nm)
                    
        elif clf_choice == '2':

            for choice in in_choice:
                dataset_nm = utils.get_key_from_value(dataset_dict, choice)
                data_in = utils.get_key_from_value(input_dict, choice)
                if data_in is None:
                    print("Invalid dataset menu selection {0}. Continuing...".format(choice))
                else:
                    print("\nRunning Naive Bayes using dataset {0}...".format(dataset_nm))
                    NB.run_sknb(data_in, dataset_nm)
            
        elif clf_choice == '3':
            print('Running SVM...')
            print('Running NaiveBayes...')
            print("Choices: ", in_choice)
            
        plt.show(block=False)
    else:
        print("INVALID SELECTION!")
