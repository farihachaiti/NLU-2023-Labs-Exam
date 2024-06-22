# Add functions or classes used for data loading and preprocessing
import os
import re
import nltk
nltk.download('movie_reviews')
nltk.download("subjectivity")
nltk.download('punkt')
from nltk.corpus import movie_reviews
import en_core_web_sm
nlp = en_core_web_sm.load()


mr = movie_reviews


from nltk.corpus import subjectivity
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants

#preprocessing datasets

analyzer = SentimentIntensityAnalyzer()
doc = subjectivity.sents()


def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:

        for line in f:
            # Use regular expression to find the sentence before the symbol
            match = re.search(r'(.*?){}'.format(re.escape('####')), line)

            if match:
                sentence_before_symbol = match.group(1).strip()
                dataset.append(sentence_before_symbol)

    return dataset

def lol2str(doc):
    # flatten & join
    return "".join([w for sent in doc for w in sent])

def extract(doc):
    aspects = []
    for sent in doc.sents:
        target = None
        opinion = None
        for tok in sent:
            if tok.dep_ == 'nsubj' and tok.pos_ == 'NOUN':
                target = tok.text
            if tok.pos_ == 'ADJ':
                descr = ''
                for child in tok.children:
                    if child.pos_ != 'ADV':
                        continue
                    descr += child.text + ' '
                opinion = descr + tok.text
        if target:
            aspects.append((target, opinion))
    return aspects


train_raw = load_data(os.path.join('dataset/laptop14_train.txt'))
test_raw = load_data(os.path.join('dataset/laptop14_test.txt'))
