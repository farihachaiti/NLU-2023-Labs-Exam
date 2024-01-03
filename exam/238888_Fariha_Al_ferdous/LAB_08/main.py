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

    vectors = vectorizer.fit_transform(data)
    lblencoder.fit(lbls)
    labels = lblencoder.transform(lbls)

    #extending collocational features
    data_col = [extend_collocational_features(inst) for inst in senseval.instances('interest.pos')]
    #results
    print(data_col[0])


    dvectorizer = DictVectorizer(sparse=False)
    dvectors = dvectorizer.fit_transform(data_col)

    #Concatenating BOW and new collocational feature vectors
    uvectors = np.concatenate((vectors.toarray(), dvectors), axis=1)

    #evaluation
    scores = cross_validate(classifier, uvectors, labels, cv=stratified_split, scoring=['f1_micro'])

    #results
    print("\033[1mEvaluation score for Concatenated BAO and Extended Collocational Feature Vectors:\033[0m")
    print("{:.3f}".format(sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])))

    #evaluating original lesk and lesk similarity

    for i, inst in enumerate(senseval.instances('interest.pos')):
        txt = [t[0] for t in inst.context]
        raw_ref = inst.senses[0] # let's get first sense
        hyp = original_lesk(txt, txt[inst.position], synsets=synsets, majority=True).name()
        hyp2 = lesk_similarity(txt, txt[inst.position], similarity="resnik", synsets=synsets, majority=True).name()
        ref = mapping.get(raw_ref)
        
        # for precision, recall, f-measure        
        refs[ref].add(i)
        hyps[hyp].add(i)
        hyps2[hyp2].add(i)
        
        
        # for accuracy
        refs_list.append(ref)
        hyps_list.append(hyp)
        hyps_list2.append(hyp2)

    print("\033[1mFor Original Lesk:\033[0m Acc:", round(accuracy(refs_list, hyps_list), 3))
    print("\033[1mFor Lesk Similarity:\033[0m Acc:", round(accuracy(refs_list, hyps_list2), 3))

    for cls in hyps.keys():
        p = precision(refs[cls], hyps[cls])
        r = recall(refs[cls], hyps[cls])
        f = f_measure(refs[cls], hyps[cls], alpha=1)
    print("\033[1mEvaluation score For Original Lesk:\033[0m p={:.3f}; r={:.3f}; f={:.3f}; s={}".format(p, r, f, len(refs[cls])))

    for cls in hyps2.keys():  
        p = precision(refs[cls], hyps2[cls])
        r = recall(refs[cls], hyps2[cls])
        f = f_measure(refs[cls], hyps2[cls], alpha=1)
    print("\033[1mEvaluation score For Lesk Similarity:\033[0m p={:.3f}; r={:.3f}; f={:.3f}; s={}".format(p, r, f, len(refs[cls])))
