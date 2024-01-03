# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer




if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space', 'misc.forsale']
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    clf = svm.SVC(kernel="linear",C=2) #taking hyperparameter C=2 to converge

    #original count vectorizer
    print("\033[1mLinear SVM Evaluation (CountVect)\033[0m")
    vectorizer = CountVectorizer()
    data = extract_features(vectorizer, categories)
    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (CountVect-binary)\033[0m")

    #taking binary of countvect
    vectorizer = CountVectorizer(binary=True)
    data = extract_features(vectorizer, categories)
    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (TFIDF)\033[0m")
    #original tfidf
    vectorizer = TfidfVectorizer()
    data = extract_features(vectorizer, categories)
    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (TFIDF-minCutOff)\033[0m")
    #taking min cutoff for tfidf
    vectorizer = TfidfVectorizer(min_df=2)
    data = extract_features(vectorizer, categories)
    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (TFIDF-maxCutOff)\033[0m")
    #taking max cutoff for tfidf
    vectorizer = TfidfVectorizer(max_df=5)
    data = extract_features(vectorizer, categories)
    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (TFIDF-WithoutStopWords)\033[0m")
    #taking stopwords to None for tfidf
    vectorizer = TfidfVectorizer(stop_words=None)
    data = extract_features(vectorizer, categories)
    SVM_evaluation(clf, data, stratified_split)

    print("\033[1mLinear SVM Evaluation (TFIDF-NoLowerCase)\033[0m")
    #using tfidf without lowercase
    vectorizer = TfidfVectorizer(lowercase=False)
    data = extract_features(vectorizer, categories)
    SVM_evaluation(clf, data, stratified_split)