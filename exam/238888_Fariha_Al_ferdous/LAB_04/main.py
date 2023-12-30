# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import nltk
from nltk.corpus import treebank
from nltk.tag import DefaultTagger 
import spacy
import math
import en_core_web_sm
from spacy.tokenizer import Tokenizer


if __name__ == "__main__":

    # Prepare Training & Test Splits as 90%/10%
    nltk.download('treebank')

    total_size = len(treebank.tagged_sents())
    train_indx = math.ceil(total_size * 0.9)
    trn_data = treebank.tagged_sents(tagset='universal')[:train_indx]
    tst_data = treebank.tagged_sents(tagset='universal')[train_indx:]

    print("\033[1mTotal:\033[0m {}; \033[1mTrain:\033[0m {}; \033[1mTest:\033[0m {}".format(total_size, len(trn_data), len(tst_data)))


    print("\033[1mTagging with NgramTagger\033[0m") 
    
    #Using DefaultTagger with cutoff 1
    backoff = DefaultTagger('NN')
    ngramTagger = nltk.NgramTagger(1,train=trn_data,cutoff=1,backoff=backoff)

    # tagging sentences in test set
    for s in treebank.sents()[train_indx:]:
        print("\033[1mINPUT:\033[0m {}".format(s))
        print("\033[1mTAG:\033[0m {}".format(ngramTagger.tag(s)))
        break

    # evaluation
    accuracy = ngramTagger.accuracy(tst_data)

    print("\033[1mAccuracy of NgramTagger:\033[0m {:6.4f}".format(accuracy))

    mapping_spacy_to_NLTK = {
        "ADJ": "ADJ",
        "ADP": "ADP",
        "ADV": "ADV",
        "AUX": "VERB",
        "CCONJ": "CONJ",
        "DET": "DET",
        "INTJ": "X",
        "NOUN": "NOUN",
        "NUM": "NUM",
        "PART": "PRT",
        "PRON": "PRON",
        "PROPN": "NOUN",
        "PUNCT": ".",
        "SCONJ": "CONJ",
        "SYM": "X",
        "VERB": "VERB",
        "X": "X"
    }


    nlp = en_core_web_sm.load()
    nlp.tokenizer = Tokenizer(nlp.vocab)

    print("\033[1mSpacy to NLTK Tag Mapping:\033[0m {}".format(mapping_spacy_to_NLTK))


    print("\033[1mTagging with Spacy\033[0m")
    
    for id_sent,sent in enumerate(treebank.sents()[train_indx:]):
        doc = nlp(" ".join(sent))
        break

    test = [(t.text, t.pos_) for t in doc]

    for s in treebank.sents()[train_indx:]:
        print("\033[1mINPUT:\033[0m {}".format(s))
        reference = ngramTagger.tag(s)
        break
        
    print("\033[1mREFERENCE:\033[0m {}".format(reference))
    print("\033[1mTAG:\033[0m {}".format(test))



    accuracy = nlp_accuracy(reference, test)
    print("\033[1mAccuracy of Spacy:\033[0m {:6.4f}".format(accuracy))

