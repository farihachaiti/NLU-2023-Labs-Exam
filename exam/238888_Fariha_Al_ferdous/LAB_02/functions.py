# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups
import numpy as np

#function to evaluate the classifier
def SVM_evaluation(clf, data, split):
    hyps_list = []
    refs_list = []
    scores_list= []
    # for each training/testing fold
    for train_index, test_index in split.split(data.data, data.target):
        # train (fit) model
        clf.fit(data.data[train_index], data.target[train_index])
        # predict test labels
        clf.predict(data.data[test_index])
        # predict test labels
        hyps = clf.predict(data.data[test_index])
        refs = data.target[test_index]
        hyps_list.append(hyps)  
        refs_list.append(refs)
    refs_list2 = [item for sublist in refs_list for item in sublist]
    hyps_list2 = [item for sublist in hyps_list for item in sublist]
    scores = cross_validate(clf, data.data, data.target, cv=split, scoring=['f1_weighted']) 

    print(classification_report(refs_list2, hyps_list2, target_names=data.target_names))
    print(sum(scores['test_f1_weighted'])/len(scores['test_f1_weighted']))

#function to extract features
def extract_features(vectorizer, categories):
    data = fetch_20newsgroups(subset='train', categories=categories)
    data.data = vectorizer.fit_transform(data.data)
    data.data.shape

    return data