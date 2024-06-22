# Add functions or classes used for data loading and preprocessing
import numpy
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn

#preprocessing datasets
pos2wn = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

def lol2str(doc):
    # flatten & join
    return "".join([w for sent in doc for w in sent])

def lesk(context_sentence, ambiguous_word, pos=None, synsets=None):
    """Return a synset for an ambiguous word in a context.

    :param iter context_sentence: The context sentence where the ambiguous word
         occurs, passed as an iterable of words.
    :param str ambiguous_word: The ambiguous word that requires WSD.
    :param str pos: A specified Part-of-Speech (POS).
    :param iter synsets: Possible synsets of the ambiguous word.
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.
    """

    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        if pos == 'a':
            synsets = [ss for ss in synsets if str(ss.pos()) in ['a', 's']]
        else:
            synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    _, sense = max(
        (len(context.intersection(ss.definition().split())), ss) for ss in synsets
    )

    return sense


def score_sent(document, use_pos=False):
    pos = []
    neg = []
    obj = []
    for sent in document:
        if use_pos:
            tagged_sent = pos_tag(sent, tagset='universal')
        else:
            tagged_sent = [(w, None) for w in sent]

        for tok, tag in tagged_sent:
            ss = lesk(sent, tok, pos=pos2wn.get(tag, None))
            if ss:
                sense = swn.senti_synset(ss.name())
                pos.append(sense.pos_score())
                neg.append(sense.neg_score())
                obj.append(sense.obj_score())
    return pos, neg, obj

def subjectivity_sentence_level(document, analyzer):
    S = 0
    O = 0
    labels = ['S', 'O']
    for sentence in document:
        value = analyzer.polarity_scores(" ".join(sentence))
        if value["compound"] != 0:
            S+=1
        else:
            O+=1
    return labels[numpy.argmax(numpy.asarray([S, O]))]
def rm_objective_sentences(document, analyzer):
    new_doc = []
    for sentence in document:
        value = analyzer.polarity_scores(" ".join(sentence))
        if value["compound"] != 0:
            new_doc.append(" ".join(sentence))
    return new_doc
def separate_objective_sentences(document, analyzer):
    new_doc = []
    obj_doc = []
    for sentence in document:
        value = analyzer.polarity_scores(" ".join(sentence))
        if value["compound"] != 0:
            new_doc.append(" ".join(sentence))
        else:
            obj_doc.append(" ".join(sentence))
    return new_doc, obj_doc
def polarity_doc_level(document, analyzer):
    value = analyzer.polarity_scores(document)
    P = 0
    N = 0
    labels = ['P', 'N']
    if value["compound"] > 0:
        P += 1
    elif value["compound"] <= 0: # In this way we penalize the neg class
        N += 1
    return labels[numpy.argmax(numpy.asarray([P, N]))]