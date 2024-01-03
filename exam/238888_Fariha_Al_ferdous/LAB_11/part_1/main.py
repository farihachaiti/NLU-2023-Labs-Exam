# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import nltk
nltk.download('movie_reviews')
nltk.download("subjectivity")
nltk.download('punkt')
from nltk.corpus import movie_reviews
from sklearn.metrics import classification_report
import spacy
import en_core_web_sm

from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pickle
nlp = en_core_web_sm.load()


mr = movie_reviews


from nltk.corpus import subjectivity
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    analyzer = SentimentIntensityAnalyzer()
    doc = subjectivity.sents()


    # Train and test with Stratified K Fold

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    scores_clf = []
    scores_subjectivity = []
    new_doc = []

    ref_sub, ref_ob = separate_objective_sentences(doc, analyzer)
    corpus = [lol2str(d) for d in ref_ob] + [lol2str(d) for d in ref_sub]
    ref = numpy.array([0] * len(ref_ob) + [1] * len(ref_sub))

    for i, (train_index, test_index) in enumerate(skf.split(corpus, ref)):
        x_train, x_test = [corpus[indx] for indx in train_index], [corpus[indx] for indx in test_index]
        y_train, y_test = [ref[indx] for indx in train_index], [ref[indx] for indx in test_index]
        # Needed for word and sentence level

        test_x_split = [[sentence.split() for sentence in corpus] for corpus in x_test]
    
        vectorizer = CountVectorizer()
        vectorizer.fit(x_train)
        train_features = vectorizer.transform(x_train)
        test_features = vectorizer.transform(x_test)
        
        clf = MLPClassifier(random_state=1, max_iter=300).fit(train_features, y_train)

        hyp = clf.predict(test_features)
        new_doc.append(hyp)
        scores_clf.append(f1_score(y_test, hyp, average='macro'))

    #saving model      
    model_filename = 'bin/mlp_classifier_model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(clf, model_file) 

    #results 
    print('\033[1mSubjectivity: F1 classifier:\033[0m', round(sum(scores_clf)/len(scores_clf), 3))
    print('\033[1mSubjectivity: Classification Report:\033[0m')
    print(classification_report(y_test, hyp))


    scores_clf = []


    rev_neg = mr.paras(categories='neg')
    rev_pos = mr.paras(categories='pos')

    rev_neg_wo_objective = ["\n".join(rm_objective_sentences(doc, analyzer)) for doc in rev_neg]
    rev_pos_wo_objective = ["\n".join(rm_objective_sentences(doc, analyzer)) for doc in rev_pos]

    corpus_wo_objective = rev_neg_wo_objective + rev_pos_wo_objective

    ref = numpy.array([0] * len(rev_neg_wo_objective) + [1] * len(rev_pos_wo_objective))

    for i, (train_index, test_index) in enumerate(skf.split(corpus_wo_objective, ref)):
        x_train, x_test = [corpus_wo_objective[indx] for indx in train_index], [corpus_wo_objective[indx] for indx in test_index]
        y_train, y_test = [ref[indx] for indx in train_index], [ref[indx] for indx in test_index]
        # Needed for word and sentence level
        test_x_split = [[sentence.split() for sentence in doc.splitlines()] for doc in x_test]

        vectorizer = CountVectorizer()
        vectorizer.fit(x_train)
        train_features = vectorizer.transform(x_train)
        test_features = vectorizer.transform(x_test)
        
        clf = MLPClassifier(random_state=1, max_iter=300).fit(train_features, y_train)
       
        hyp = clf.predict(test_features)
        
        scores_clf.append(f1_score(y_test, hyp, average='macro'))

    #saving model
    model_filename = 'bin/mlp_classifier_model_2.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(clf, model_file)  

    #results 
    print('\033[1mF1 classifier Polarity w/o Obj:\033[0m', round(sum(scores_clf)/len(scores_clf), 3))
    print('\033[1mPolarity w/o Obj: Classification Report:\033[0m')
    print(classification_report(y_test, hyp))


    scores_clf = []


    rev_neg = mr.paras(categories='neg')
    rev_pos = mr.paras(categories='pos')
    corpus = [lol2str(d) for d in rev_neg] + [lol2str(d) for d in rev_pos]


    ref = numpy.array([0] * len(rev_neg) + [1] * len(rev_pos))

    for i, (train_index, test_index) in enumerate(skf.split(corpus, ref)):
        x_train, x_test = [corpus[indx] for indx in train_index], [corpus[indx] for indx in test_index]
        y_train, y_test = [ref[indx] for indx in train_index], [ref[indx] for indx in test_index]
        # Needed for word and sentence level
        test_x_split = [[sentence.split() for sentence in doc.splitlines()] for doc in x_test]

        vectorizer = CountVectorizer()
        vectorizer.fit(x_train)
        train_features = vectorizer.transform(x_train)
        test_features = vectorizer.transform(x_test)
        
        clf = MLPClassifier(random_state=1, max_iter=300).fit(train_features, y_train)

        hyp = clf.predict(test_features)
        scores_clf.append(f1_score(y_test, hyp, average='macro'))
        


    #saving model
    model_filename = 'bin/mlp_classifier_model_3.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(clf, model_file)  

    #results     
    print('\033[1mF1 classifier Polarity with Obj:\033[0m', round(sum(scores_clf)/len(scores_clf), 3))
    print('\033[1mPolarity with Obj: Classification Report:\033[0m')
    print(classification_report(y_test, hyp))