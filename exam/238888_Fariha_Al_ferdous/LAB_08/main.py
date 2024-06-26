# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from nltk.metrics.scores import precision, recall, f_measure, accuracy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import numpy as np
nltk.download('senseval')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import senseval


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results


    # Let's create mapping from convenience
    #preprocessing
    mapping = {
        'interest_1': 'interest.n.01',
        'interest_2': 'interest.n.03',
        'interest_3': 'pastime.n.01',
        'interest_4': 'sake.n.01',
        'interest_5': 'interest.n.05',
        'interest_6': 'interest.n.04',
    }

    refs = {k: set() for k in mapping.values()}
    hyps = {k: set() for k in mapping.values()}
    hyps2 = {k: set() for k in mapping.values()}
    refs_list = []
    hyps_list = []
    hyps_list2 = []

    # since WordNet defines more senses, let's restrict predictions

    synsets = []
    for ss in wordnet.synsets('interest', pos='n'):
        if ss.name() in mapping.values():
            # You need to preporecess the definitions
            # Give a look at the preprocessing function that we defined above 
            defn = ss.definition()
            # let's use the same preprocessing
            tags = preprocess(defn)
            toks = [l for w, l, p in tags]
            synsets.append((ss,toks))
   

    vectorizer = CountVectorizer()
    classifier = MultinomialNB()
    lblencoder = LabelEncoder()
    data = [" ".join([t[0] for t in inst.context]) for inst in senseval.instances('interest.pos')]
    lbls = [inst.senses[0] for inst in senseval.instances('interest.pos')]
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)  
    lblencoder.fit(lbls)
    labels = lblencoder.transform(lbls)
    vectors = vectorizer.fit_transform(data)
    
    #Extending collocational feature vectors
    print("\033[1mEvaluation score for Extended Collocational Feature Vectors:\033[0m")
    #extending collocational features
    data_col = [extend_collocational_features(inst) for inst in senseval.instances('interest.pos')]
    dvectorizer = DictVectorizer(sparse=False)
    dvectors = dvectorizer.fit_transform(data_col)
    #evaluation
    scores = cross_validate(classifier, dvectors, labels, cv=stratified_split, scoring=['f1_micro'])

    #results
    print("{:.3f}".format(sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])))
    

    #Concatenating BOW and new collocational feature vectors
    print("\033[1mEvaluation score for Concatenated BAO and Extended Collocational Feature Vectors:\033[0m")


    uvectors = np.concatenate((vectors.toarray(), dvectors), axis=1)

    #evaliuation 
    scores = cross_validate(classifier, uvectors, labels, cv=stratified_split, scoring=['f1_micro'])
     #results
    print("{:.3f}".format(sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])))

    #evaluating original lesk and lesk similarity\
    

    for i, inst in enumerate(senseval.instances('interest.pos')):
        txt = [t[0] for t in inst.context]
        hyp = original_lesk(txt, txt[inst.position], synsets=synsets, majority=True).name()
        hyp2 = lesk_similarity(txt, txt[inst.position], similarity="resnik", synsets=synsets, majority=True).name()
        
        # for precision, recall, f-measure        
        hyps[hyp].add(i)
        hyps2[hyp2].add(i)
        
        
        # for accuracy
        hyps_list.append(hyp)
        hyps_list2.append(hyp2)

    print("\033[1mEvaluation score for Original Lesk:\033[0m")
    #evaluation
    scores = cross_validate(classifier, vectors, hyps_list, cv=stratified_split, scoring=['f1_micro'])

    #results
    print("{:.3f}".format(sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])))

    print("\033[1mEvaluation score for Lesk Similarity:\033[0m")
    #evaluation
    scores = cross_validate(classifier, vectors, hyps_list2, cv=stratified_split, scoring=['f1_micro'])

    #results
    print("{:.3f}".format(sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])))

    print("\033[1mAs we can see, Accuracy for Less Similarity is less than the one for Original Lesk\033[0m")

    