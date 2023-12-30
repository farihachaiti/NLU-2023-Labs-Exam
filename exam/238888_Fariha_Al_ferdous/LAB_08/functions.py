# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk
from nltk.corpus import wordnet_ic
nltk.download('wordnet_ic')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.util import ngrams

# A bit of preprocessing 
def preprocess(text):
    mapping = {"NOUN": wordnet.NOUN, "VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV}
    sw_list = ['on', 'the', 'of', 'a']
    
    lem = WordNetLemmatizer()
    # tokenize, if input is text
    tokens = nltk.word_tokenize(text) if type(text) is str else text
    # compute pos-tag
    tagged = nltk.pos_tag(tokens, tagset='universal') # nltk.pos_tag(, tagset="universal")
    # lowercase
    tagged = [(w.lower(), p) for w, p in tagged]
    # optional: remove all words that are not NOUN, VERB, ADJ, or ADV (i.e. no sense in WordNet)
    tagged = [(w, p) for w, p in tagged if p in mapping]
    # re-map tags to WordNet (return orignal if not in-mapping, if above is not used)
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    # remove stopwords
    tagged = [(w, p) for w, p in tagged if w not in sw_list] #... not in stopword list]
    # lemmatize
    tagged = [(w, lem.lemmatize(w, pos=p), p) for w, p in tagged]
    # unique the list
    tagged = list(set(tagged))
    
    return tagged
def get_sense_definitions(context):
    # input is text or list of strings
    lemma_tags = preprocess(context)

    # let's get senses for each
    senses = [(w, wordnet.synsets(l, p)) for w, l, p in lemma_tags]

    # let's get their definitions
    definitions = []
    for raw_word, sense_list in senses:
        if len(sense_list) > 0:
            # let's tokenize, lowercase & remove stop words 
            def_list = []
            for s in sense_list:
                defn = s.definition()
                # let's use the same preprocessing
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                def_list.append((s, toks))
            definitions.append((raw_word, def_list))
    return definitions
    
def get_top_sense(words, sense_list):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    val, sense = max((len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list)
    return val, sense

# Lesk simplified
def lesk_simplified(context_sentence, ambiguous_word, pos=None, synsets=None):

    context = set(context_sentence)
    
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)
    # Filter by pos-tag
    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None
    
    _, sense = max((len(set(context).intersection(set(nltk.word_tokenize(ss.definition())))), ss) for ss in synsets)  # Don't forget to tokenize the definition
    
    return sense


def get_top_sense_sim(context_sense, sense_list, similarity):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                scores.append((context_sense.path_similarity(ss), ss))
            except:
                scores.append((0, ss))    
        elif similarity == "lch":
            try:
                scores.append((context_sense.lch_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                scores.append((context_sense.wup_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "resnik":
            try:
                scores.append((context_sense.res_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                scores.append((context_sense.lin_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "jiang":
            try:
                scores.append((context_sense.jcn_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None
    val, sense = max(scores)
    return val, sense


def lesk_similarity(context_sentence, ambiguous_word, similarity="resnik", pos=None, 
                    synsets=None, majority=True):
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))
    
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    scores = []
    
    # Here you may have some room for improvement
    # For instance instead of using all the definitions from the context
    # you pick the most common one of each word (i.e. the first)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))
            
    if len(scores) == 0:
        return synsets[0][0]
    
    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    
    return best_sense

def original_lesk(context_sentence, ambiguous_word, pos=None, synsets=None, majority=False):

    context_senses = get_sense_definitions(set(context_sentence)-set([ambiguous_word]))
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    scores = []
    # print(synsets)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense(sense[1], synsets))
            
    if len(scores) == 0:
        return synsets[0][0]
    
    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    return best_sense


def extend_collocational_features(inst):
    p = inst.position
    return {
        "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
        "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
        "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
        "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0],
        "pos-tag for w-2_word": 'NULL' if p < 2 else nltk.pos_tag(inst.context[p-2], tagset="universal")[1][1],
        "pos-tag for w-1_word": 'NULL' if p < 1 else nltk.pos_tag(inst.context[p-1], tagset="universal")[1][1],
        "pos-tag for w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else nltk.pos_tag(inst.context[p+1], tagset="universal")[1][1],
        "pos-tag for w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else nltk.pos_tag(inst.context[p+2], tagset="universal")[1][1],
        "ngrams within w-2": 'NULL' if p < 2 else list(ngrams(inst.context[p-2], 2))[0],
        "ngrams within w-1": 'NULL' if p < 1 else list(ngrams(inst.context[p-1], 2))[0],
        "ngrams within w+1": 'NULL' if len(inst.context) - 1 < p+1 else list(ngrams(inst.context[p+1], 2))[0],
        "ngrams within w+2": 'NULL' if len(inst.context) - 1 < p+2 else list(ngrams(inst.context[p+2], 2))[0]
    }