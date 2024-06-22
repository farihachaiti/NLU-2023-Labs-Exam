# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pickle

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    # Train and test with Stratified K Fold

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    scores_clf = []

    #Task 1: Extraction of Aspect Terms

    # Extracting Aspect terms
    aspects_train = extract(nlp(' '.join(train_raw)))
    aspects_test = extract(nlp(' '.join(test_raw)))

    # Replace None with 'None' in aspects_train and aspects_test
    aspects_train_fixed = [(target, 'None' if opinion is None else opinion) for target, opinion in aspects_train]
    aspects_test_fixed = [(target, 'None' if opinion is None else opinion) for target, opinion in aspects_test]


    half_len = len(aspects_train_fixed) // 2

    ref_trn = numpy.array([0] * half_len + [1] * half_len)



    half_len = len(aspects_test_fixed) // 2

    ref_tst = numpy.array([0] * half_len + [1] * half_len)

    for i, ((train_index, train_index), (test_index, test_index)) in enumerate(zip(skf.split(aspects_train_fixed, ref_trn), skf.split(aspects_test_fixed, ref_tst))):

        x_train, x_test = [' '.join(aspects_train_fixed[idx]) for idx in train_index if aspects_train[idx] is not None], [' '.join(aspects_test_fixed[idx]) for idx in test_index if aspects_test[idx] is not None]
        y_train, y_test = [ref_trn[idx] for idx in train_index], [ref_tst[idx] for idx in test_index]


        vectorizer = CountVectorizer()
        vectorizer.fit(x_train)
        train_features = vectorizer.transform(x_train)
        test_features = vectorizer.transform(x_test)

        clf = MLPClassifier(random_state=1, max_iter=300).fit(train_features, y_train)

        hyp = clf.predict(test_features)
        scores_clf.append(f1_score(y_test, hyp, average='macro'))

    #saving model
    model_filename = 'bin/mlp_classifier_model_extract_aspects.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(clf, model_file)  

    #results 
    print('\033[1mF1 classifier Extraction of aspect terms:\033[0m', round(sum(scores_clf)/len(scores_clf), 3))
    print('\033[1mExtraction of Aspect terms: Classification Report\033[0m:')
    print(classification_report(y_test, hyp))


    #Task 2: Polarity of Aspect Terms


    rev_t_pos, rev_o_pos, rev_t_neg, rev_o_neg = polarity_aspect(aspects_train, analyzer)
    corpus_trn = [lol2str(d) for d in rev_t_neg] + [lol2str(d) for d in rev_t_pos] + [lol2str(d) for d in rev_o_neg] + [lol2str(d) for d in rev_o_pos]

    ref_trn = numpy.array([0] * len(rev_t_neg) + [1] * len(rev_t_pos) + [2] * len(rev_o_neg) + [3] * len(rev_o_pos))

    rev_t_pos2, rev_o_pos2, rev_t_neg2, rev_o_neg2 = polarity_aspect(aspects_test, analyzer)
    corpus_tst = [lol2str(d) for d in rev_t_neg2] + [lol2str(d) for d in rev_t_pos2] + [lol2str(d) for d in rev_o_neg2] + [lol2str(d) for d in rev_o_pos2]


    ref_tst = numpy.array([0] * len(rev_t_neg2) + [1] * len(rev_t_pos2) + [2] * len(rev_o_neg2) + [3] * len(rev_o_pos2))



    for i, ((train_index, train_index), (test_index, test_index)) in enumerate(zip(skf.split(corpus_trn, ref_trn), skf.split(corpus_tst, ref_tst))):

        x_train, x_test = [corpus_trn[idx] for idx in train_index], [corpus_tst[idx] for idx in test_index]
        y_train, y_test = [ref_trn[idx] for idx in train_index], [ref_tst[idx] for idx in test_index]


        vectorizer = CountVectorizer()
        vectorizer.fit(x_train)
        train_features = vectorizer.transform(x_train)
        test_features = vectorizer.transform(x_test)

        clf = MLPClassifier(random_state=1, max_iter=300).fit(train_features, y_train)
   
        hyp = clf.predict(test_features)
        scores_clf.append(f1_score(y_test, hyp, average='macro'))

    #saving model
    model_filename = 'bin/mlp_classifier_model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(clf, model_file)  




    #results 
    print('\033[1mF1 classifier Polarity of aspect terms:\033[0m', round(sum(scores_clf)/len(scores_clf), 3))
    print('\033[1mPolarity of Aspect terms: Classification Report\033[0m:')
    print(classification_report(y_test, hyp))


    print('\033[1mSEMEVAL Polarity of aspect terms:\033[0m')
    print(aspect_polarity_estimation(y_test, hyp))
