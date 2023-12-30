# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from sklearn.datasets import fetch_20newsgroups
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer




if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space', 'misc.forsale']
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    clf = svm.SVC(kernel="linear",C=2)


    print("\033[1mLinear SVM Evaluation (CountVect)\033[0m")
    vectorizer = CountVectorizer()
    data = fetch_20newsgroups(subset='train', categories=categories)
    data.data = vectorizer.fit_transform(data.data)
    data.data.shape
    

    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (CountVect-binary)\033[0m")

    vectorizer = CountVectorizer(binary=False)
    data = fetch_20newsgroups(subset='train', categories=categories)
    data.data = vectorizer.fit_transform(data.data)
    data.data.shape

    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (TFIDF)\033[0m")

    vectorizer = TfidfVectorizer()
    data = fetch_20newsgroups(subset='train', categories=categories)
    data.data = vectorizer.fit_transform(data.data)
    data.data.shape

    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (TFIDF-minCutOff)\033[0m")

    vectorizer = TfidfVectorizer(min_df=2)
    data = fetch_20newsgroups(subset='train', categories=categories)
    data.data = vectorizer.fit_transform(data.data)
    data.data.shape

    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (TFIDF-maxCutOff)\033[0m")

    vectorizer = TfidfVectorizer(max_df=5)
    data = fetch_20newsgroups(subset='train', categories=categories)
    data.data = vectorizer.fit_transform(data.data)
    data.data.shape

    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (TFIDF-WithoutStopWords)\033[0m")

    vectorizer = TfidfVectorizer(stop_words=None)
    data = fetch_20newsgroups(subset='train', categories=categories)
    data.data = vectorizer.fit_transform(data.data)
    data.data.shape

    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (TFIDF-NoLowerCase)\033[0m")

    vectorizer = TfidfVectorizer(lowercase=False)
    data = fetch_20newsgroups(subset='train', categories=categories)
    data.data = vectorizer.fit_transform(data.data)
    data.data.shape

    SVM_evaluation(clf, data, stratified_split)