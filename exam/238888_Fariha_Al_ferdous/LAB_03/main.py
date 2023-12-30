# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from nltk.lm.preprocessing import flatten
from nltk.lm import Vocabulary
from nltk.probability import FreqDist
from nltk.lm.preprocessing import padded_everygram_pipeline
from itertools import chain
from nltk.corpus import gutenberg

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    macbeth_sents = [[w.lower() for w in sent] for sent in gutenberg.sents('shakespeare-macbeth.txt')]
    macbeth_words = flatten(macbeth_sents)
    # Compute vocab 
    lex = Vocabulary(macbeth_words, unk_cutoff=2)
    # Handeling OOV
    macbeth_oov_sents = [list(lex.lookup(sent)) for sent in macbeth_sents]
    padded_ngrams_oov, flat_text_oov = padded_everygram_pipeline(2, macbeth_oov_sents)
    # Train the model 
    fdist = FreqDist(chain.from_iterable(macbeth_oov_sents))

    #Custom StupidBackoff algorithm with alpha=0.4
    stupid = MyStupidBackoff(macbeth_oov_sents, fdist, 2, 0.4)
    stupid.fit(padded_ngrams_oov, flat_text_oov)

    ngrms, flat_text = padded_everygram_pipeline(stupid.order, macbeth_sents)
    ngrms = chain.from_iterable(ngrms)

    print("\033[1mMyStupidBackoff StupidBackoff : Perplexity\033[0m")
    print("{:.3}".format(stupid.perplexity([x for x in ngrms if len(x) == stupid.order])))

    print("\033[1mMyStupidBackoff StupidBackoff : Manual Function Perplexity\033[0m")
    print("{:.3}".format(stupid.compute_ppl(stupid, macbeth_sents)))