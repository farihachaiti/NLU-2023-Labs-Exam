# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from nltk.lm.api import LanguageModel
from nltk import word_tokenize 
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import NgramCounter
from itertools import chain
import math
import numpy as np

#custom stupid backoff algorithm
class MyStupidBackoff(LanguageModel):
    
    def __init__(self, context, fdist, order, alpha=0.4, *args, **kwargs):
        super().__init__(order, *args, **kwargs)
        self.alpha = alpha
        self.context = context
        self.fdist = fdist
        self.order = order


    def compute_ngram(self, order, text):
        token = word_tokenize(text)
        ngram_words = list(ngrams(token, order))
        return ngram_words

    #manual perplexity calculating function
    def compute_ppl(self, model, data):
        highest_ngram = model.order
        scores = []
        for sentence in data:   
            ngrams, flat_text = padded_everygram_pipeline(highest_ngram, [sentence])
            ngrams = chain.from_iterable(ngrams)
            scores.extend([-1 * model.logscore(w[-1], w[0:-1]) for w in ngrams if len(w) == highest_ngram])
        return math.pow(2.0, np.asarray(scores).mean())

    def stupid_backoff(self, word, context=None):              
        padded_ngrams, flat_text = padded_everygram_pipeline(self.order, context)
        unigrams = self.compute_ngram(1, word)
        if not context:
            return len(unigrams) 
        if self.fdist.freq(word)<=0:
            return self.alpha * self.stupid_backoff(word, context[1:])
        else: 
            return self.fdist.freq(word)/NgramCounter(padded_ngrams).N()

    def unmasked_score(self, word, context=None):
        return self.stupid_backoff(word, context)