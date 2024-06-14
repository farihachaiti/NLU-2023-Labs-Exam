# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from collections import Counter
import nltk

#function to print statistics
def statistics(chars, words, sents):
    word_lens = [len(word) for word in words]
    sent_lens = [len(sent) for sent in sents]
    chars_in_sents = [len(''.join(sent)) for sent in sents]
    
    word_per_sent = round(sum(sent_lens) / len(sents))
    char_per_word = round(sum(word_lens) / len(words))
    char_per_sent = round(sum(chars_in_sents) / len(sents))
    
    longest_sentence = max(sent_lens)
    longest_word = max(word_lens)
    
    print('Word per sentence', word_per_sent)
    print('Char per word', char_per_word)
    print('Char per sentence', char_per_sent)
    print('Longest sentence', longest_sentence)
    print('Longest word', longest_word)

#cutoff function
def cut_off(vocab, n_min=100, n_max=100):
    new_vocab = []
    for word, count in vocab.items():
        if count >= n_min and count <= n_max:
            new_vocab.append(word)
    return new_vocab

#function to get nbest words
def nbest(d, n=1):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])

def print_lowercased_lexicons(corp):
    lex = set([w.lower() for w in corp])
    print(len(lex))


#function to get frequency distribution
def frequency_distribution_ref(corp):
    ref_freq_list = Counter(corp)
    best_words = nbest(ref_freq_list,n=5)
    print(best_words)

def frequency_distribution_nltk(corp):
    nltk_freq_list = nltk.FreqDist(corp)
    best_words = nbest(nltk_freq_list,n=5)
    print(best_words)

def frequency_distribution_spacy(corp):
    spacy_freq_list = Counter(corp)
    best_words = nbest(spacy_freq_list,n=5)
    print(best_words)