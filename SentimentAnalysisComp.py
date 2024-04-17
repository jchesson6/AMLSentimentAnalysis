import subprocess

menu_map = {'SVM': 1, 'NB': 2, 'Both': 3, 'Exit': 4}
input_map = {'inputs/Tweets.csv': 1, 'inputs/twcs.csv': 2, 'inputs/IMDBDataset.csv': 3}

while True:
    print("\nCLASSIFIER MENU")
    print("------------------------------------")
    print("1 - Run SVM classifier")
    print("2 - Run Naive Bayes classifier")
    print("3 - Run both classifiers for comparison")
    print("4 - Exit\n")
    clf_choice = input('Enter the menu option number you wish to choose: \n')
    if clf_choice == '4':
        break
    elif clf_choice == '1' or clf_choice == '2' or clf_choice == '3':
        print("\nDATASET MENU")
        print("------------------------------------")
        print("1 - Airline Tweets")
        print("2 - General Customer Service Tweets")
        print("3 - IMDB Review\n")
        in_choice = list(map(int, input(
            'Enter the datasets you wish to use in order separated by space only: \n').split()))
        if clf_choice == '1':
            print('Running SVM...')
            print("Choices: ", in_choice)
        elif clf_choice == '2':
            print('Running NaiveBayes...')
            print("Choices: ", in_choice)
        elif clf_choice == '3':
            print('Running SVM...')
            print('Running NaiveBayes...')
            print("Choices: ", in_choice)
    else:
        print("INVALID SELECTION!")
