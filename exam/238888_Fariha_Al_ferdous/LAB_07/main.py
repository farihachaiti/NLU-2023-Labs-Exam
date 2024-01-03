# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import nltk
from nltk.corpus import conll2002
nltk.download('conll2002')


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    print("\033[1mconll2002 Corpus\033[0m")
    print(conll2002.iob_sents('esp.train')[0])

    #performing variable features and train and testing CRF model on these
    print("\033[1mBaseline using the features in sent2spacy_features:\033[0m")
    # let's get only word and iob-tag
    trn_sents = [[(text, pos, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]
    trn_label = [sent2labels(s) for s in trn_sents]
    trn_feats = [sent2spacy_features(s) for s in trn_sents]

    tst_sents = [[(text, pos, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.testa')]
    tst_feats = [sent2spacy_features(s) for s in tst_sents]

    train_and_test_crf_model(trn_label, trn_feats, tst_sents, tst_feats)

    print("\033[1mAdd the 'suffix' feature:\033[0m")

    trn_feats = [sent2spacy_features_with_suffix(s) for s in trn_sents]
    tst_feats = [sent2spacy_features_with_suffix(s) for s in tst_sents]

    train_and_test_crf_model(trn_label, trn_feats, tst_sents, tst_feats)

    print("\033[1mAdd all the features used in the tutorial on CoNLL dataset:\033[0m")

    trn_feats = [sent2features(s) for s in trn_sents]
    tst_feats = [sent2features(s) for s in tst_sents]

    train_and_test_crf_model(trn_label, trn_feats, tst_sents, tst_feats)


    print("\033[1mwith feature window [-1, +1]:\033[0m")

    trn_feats = [sent2features_for_extended_window(s) for s in trn_sents]
    tst_feats = [sent2features_for_extended_window(s) for s in tst_sents]

    train_and_test_crf_model(trn_label, trn_feats, tst_sents, tst_feats)

    print("\033[1mwith feature window [-2, +2]:\033[0m")

    trn_feats = [sent2features_for_doubled_window(s) for s in trn_sents]
    tst_feats = [sent2features_for_doubled_window(s) for s in tst_sents]

    train_and_test_crf_model(trn_label, trn_feats, tst_sents, tst_feats)