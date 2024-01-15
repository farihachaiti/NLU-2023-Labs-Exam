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
    
    #preprocessing
    macbeth_sents = [[w.lower() for w in sent] for sent in gutenberg.sents('shakespeare-macbeth.txt')]
    macbeth_flattened = str(flatten(macbeth_sents))
    macbeth_words = word_tokenize(macbeth_flattened)
    
    # Compute vocab 
    lex = Vocabulary(macbeth_words, unk_cutoff=2)
    # Handling OOV
    macbeth_oov_sents = [list(lex.lookup(sent)) for sent in macbeth_sents]
    
   
    #Custom StupidBackoff algorithm with alpha=0.4
    order = 2
    fdist = FreqDist(chain.from_iterable(macbeth_oov_sents))
    token = word_tokenize(str(macbeth_oov_sents))
    ngram_words = list(ngrams(token, order))
   
    if not ngram_words or order==1:
        print('Please choose a higher order')
    else:        
        stupid = MyStupidBackoff(macbeth_oov_sents, fdist, order, 0.4)
        # Train the model 
        padded_ngrms_oov, flat_text_oov = padded_everygram_pipeline(stupid.order, macbeth_oov_sents)

        stupid.fit(padded_ngrms_oov, flat_text_oov)

        ngrams, flat_text = padded_everygram_pipeline(stupid.order, [lex.lookup(sent)for sent in macbeth_sents])
        ngrams = chain.from_iterable(ngrams)

        print("\033[1mMyStupidBackoff StupidBackoff : Perplexity\033[0m")
        print(stupid.perplexity([x for x in ngrams if len(x) == stupid.order]))

        print("\033[1mMyStupidBackoff StupidBackoff : Manual Function Perplexity\033[0m")
        print(stupid.compute_ppl(stupid, macbeth_sents))
    
        ngrams, flat_text = padded_everygram_pipeline(stupid.order, [lex.lookup(sent) for sent in macbeth_sents])
        ngrams = chain.from_iterable(ngrams)
        cross_entropy = stupid.cross_entropy(ngrams)
        print('\033[1mCross Entropy\033[0m:')
        print(cross_entropy)




    