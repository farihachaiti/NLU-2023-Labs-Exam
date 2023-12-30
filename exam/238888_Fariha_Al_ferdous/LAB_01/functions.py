# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
def statistics(chars, words, sents):
    word_lens = [len(word) for word in words]
    sent_lens = [len(sent) for sent in sents]
    char_lens = [len(char) for char in format(list(chars))]
    
    word_per_sent = round(sum(sent_lens) / len(sents))
    char_per_word = round(sum(char_lens) / len(words))
    char_per_sent = round(sum(char_lens) / len(sents))
    
    longest_sentence = max(sent_lens)
    longest_word = max(word_lens)
    
    print('Word per sentence', word_per_sent)
    print('Char per word', char_per_word)
    print('Char per sentence', char_per_sent)
    print('Longest sentence', longest_sentence)
    print('Longest word', longest_word)


def cut_off(vocab, n_min=100, n_max=100):
    new_vocab = []
    for word, count in vocab.items():
        if count >= n_min and count <= n_max:
            new_vocab.append(word)
    return new_vocab

def nbest(d, n=1):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])

def print_lowercased_lexicons(lex):
    print(len(lex))
    print('ALL' in lex)
    print('All' in lex)
    print('all' in lex)

def frequency_distribution(corp, lower_bound, upper_bound):
    lexicon_cut_off = len(cut_off(corp, n_min=lower_bound, n_max=upper_bound))

    print(corp.get('ALL', 0))
    print(corp.get('All', 0))
    print(corp.get('all', 0))

    print('Original', len(corp))
    print('CutOFF Min:', lower_bound, 'MAX:', upper_bound, ' Lexicon Size:', lexicon_cut_off)

    best_words = nbest(corp,n=5)
    print(best_words)
