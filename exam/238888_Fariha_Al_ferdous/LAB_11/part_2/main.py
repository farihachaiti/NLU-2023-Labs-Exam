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

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    scores_clf = []


    rev_t_pos, rev_o_pos, rev_t_neg, rev_o_neg = polarity_aspect(aspects_train, analyzer)
    corpus_trn = [lol2str(d) for d in rev_t_neg] + [lol2str(d) for d in rev_t_pos] + [lol2str(d) for d in rev_o_neg] + [lol2str(d) for d in rev_o_pos]
    #vectors = vectorizer.fit_transform(mr)
    ref_trn = numpy.array([0] * len(rev_t_neg) + [1] * len(rev_t_pos) + [2] * len(rev_o_neg) + [3] * len(rev_o_pos))

    rev_t_pos2, rev_o_pos2, rev_t_neg2, rev_o_neg2 = polarity_aspect(aspects_test, analyzer)
    corpus_tst = [lol2str(d) for d in rev_t_neg2] + [lol2str(d) for d in rev_t_pos2] + [lol2str(d) for d in rev_o_neg2] + [lol2str(d) for d in rev_o_pos2]
    #vectors = vectorizer.fit_transform(mr)

    ref_tst = numpy.array([0] * len(rev_t_neg2) + [1] * len(rev_t_pos2) + [2] * len(rev_o_neg2) + [3] * len(rev_o_pos2))

    #for (train_index, test_index) in enumerate(skf.split(corpus, ref)):
    #for i, (train_index, test_index) in enumerate(skf.split(corpus_trn, ref_trn), skf.split(corpus_tst, ref_tst)):
    for i, ((train_index, train_index), (test_index, test_index)) in enumerate(zip(skf.split(corpus_trn, ref_trn), skf.split(corpus_tst, ref_tst))):

        x_train, x_test = [corpus_trn[idx] for idx in train_index], [corpus_tst[idx] for idx in test_index]
        y_train, y_test = [ref_trn[idx] for idx in train_index], [ref_tst[idx] for idx in test_index]

        # Needed for word and sentence level
        #test_x_split = [[sentence for sentence in doc] for doc in x_test]

        #x_train = ' '.join(x_train[0])
        #x_test = [' '.join(doc) for doc in x_test]
        #unequal_indices = numpy.where(x_train != y_train)

        # Increase elements in array1 at unequal indices
        #array1[unequal_indices] += 1
        vectorizer = CountVectorizer()
        vectorizer.fit(x_train)
        train_features = vectorizer.transform(x_train)
        test_features = vectorizer.transform(x_test)

        clf = MLPClassifier(random_state=1, max_iter=300).fit(train_features, y_train)
        #scores = cross_validate(clf, vectors, ref, cv=StratifiedKFold(n_splits=10) , scoring=['f1_micro'])
        hyp = clf.predict(test_features)
        scores_clf.append(f1_score(y_test, hyp, average='macro'))

        #hyp_polarity = [polarity_doc_level(doc, analyzer) for doc in asp_doc_test]
        #scores_polarity.append(f1_score(y_test, hyp_polarity, average='macro'))

        
    print('\033[1mF1 classifier Polarity of aspect terms:\033[0m', round(sum(scores_clf)/len(scores_clf), 3))
    print('\033[1mPolarity of Aspect terms: Classification Report\033[0m:')
    print(classification_report(y_test, hyp))


    print('\033[1mSEMEVAL Polarity of aspect terms:\033[0m')
    print(aspect_polarity_estimation(y_test, hyp))
    #print(evaluate(y_test, y_test, hyp, hyp))
    #print(evaluate(ref_op, ref_targ, hyp_op, hyp_targ))