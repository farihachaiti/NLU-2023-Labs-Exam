# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
def statistics(chars, words, sents):
    word_lens = [len(word) for word in words]
    sent_lens = [len(sent) for sent in sents]
    chars_in_sents = [len(''.join(sent)) for sent in sents]
    
    word_per_sent = round(sum(sent_lens) / len(sents))
    char_per_word = round(sum(word_lens) / len(words))
    char_per_sent = round(sum(chars_in_sents) / len(sents))
    
    longest_sentence = max(sent_lens)
    longest_word = max(word_lens)
    
    return word_per_sent, char_per_word, char_per_sent, longest_sentence, longest_word


def cut_off(vocab, n_min=100, n_max=100):
    new_vocab = []
    for word, count in vocab.items():
        if count >= n_min and count <= n_max:
            new_vocab.append(word)
    return new_vocab

def nbest(d, n=1):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])
