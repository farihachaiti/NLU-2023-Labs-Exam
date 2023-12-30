# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import nltk
import spacy
import en_core_web_sm
from collections import Counter

nltk.download('gutenberg')
nltk.download('punkt')


nlp = spacy.load("en_core_web_sm")





def main():
    
    #Another Corpus
    print("\033[1mAnother Corpus\033[0m")
    milton_chars = nltk.corpus.gutenberg.raw('milton-paradise.txt')
    milton_words = nltk.corpus.gutenberg.words('milton-paradise.txt')
    milton_sents = nltk.corpus.gutenberg.sents('milton-paradise.txt')

    milton_words_nltk = nltk.word_tokenize(milton_chars)
    milton_sents_nltk = nltk.sent_tokenize(milton_chars)

    txt = milton_chars

    milton_freq_list = Counter(milton_words)

    #Descriptive statistics on the reference sentences and tokens
    word_per_sent, char_per_word, char_per_sent, longest_sentence, longest_word = statistics(milton_chars, milton_words, milton_sents)
    
    
    print("\033[1mDescriptive statistics on the reference sentences and tokens\033[0m")
    print('Word per sentence', word_per_sent)
    print('Char per word', char_per_word)
    print('Char per sentence', char_per_sent)
    print('Longest sentence', longest_sentence)
    print('Longest word', longest_word)



    # process the document
    doc = nlp(txt)

    #Descriptive statistics on the automatically processed corpus by spacy
    word_per_sent, char_per_word, char_per_sent, longest_sentence, longest_word = statistics(txt, doc, format(list(doc.sents)))

    print("\033[1mDescriptive statistics on the automatically processed corpus by spacy\033[0m")

    print('Word per sentence', word_per_sent)
    print('Char per word', char_per_word)
    print('Char per sentence', char_per_sent)
    print('Longest sentence', longest_sentence)
    print('Longest word', longest_word)

    #Descriptive statistics on the automatically processed corpus by nltk
    word_per_sent, char_per_word, char_per_sent, longest_sentence, longest_word = statistics(milton_chars, milton_words_nltk, milton_sents_nltk)

    print("\033[1mDescriptive statistics on the automatically processed corpus by nltk\033[0m")

    print('Word per sentence', word_per_sent)
    print('Char per word', char_per_word)
    print('Char per sentence', char_per_sent)
    print('Longest sentence', longest_sentence)
    print('Longest word', longest_word)

    #Lowercased lexicons for reference
    milton_lexicon = set([w.lower() for w in milton_words])

    print("\033[1mLowercased lexicons for reference\033[0m")

    print(len(milton_lexicon))
    print('ALL' in milton_lexicon)
    print('All' in milton_lexicon)
    print('all' in milton_lexicon)

    #Lowercased lexicons for spacy
    milton_lexicon_spacy = set([w.lower() for w in format(doc)])

    print("\033[1mLowercased lexicons for spacy\033[0m")

    print(len(milton_lexicon_spacy))
    print('ALL' in milton_lexicon)
    print('All' in milton_lexicon)
    print('all' in milton_lexicon)

    #Lowercased lexicons for nltk
    milton_lexicon_nltk = set([w.lower() for w in milton_words_nltk])

    print("\033[1mLowercased lexicons for nltk\033[0m")

    print(len(milton_lexicon_nltk))
    print('ALL' in milton_lexicon)
    print('All' in milton_lexicon)
    print('all' in milton_lexicon)


    #Frequency Distribution for reference
    print("\033[1mFrequency Distribution for reference\033[0m")

    print(milton_freq_list.get('ALL', 0))
    print(milton_freq_list.get('All', 0))
    print(milton_freq_list.get('all', 0))

    lower_bound = float(96) # Change these two number to compute the required cut offs
    upper_bound = float(604)
    lexicon_cut_off = len(cut_off(milton_freq_list, n_min=lower_bound, n_max=upper_bound))

    print('Original', len(milton_freq_list))
    print('CutOFF Min:', lower_bound, 'MAX:', upper_bound, ' Lexicon Size:', lexicon_cut_off)

    best_words = nbest(milton_freq_list,n=5)
    print(best_words)

    #Frequency Distribution for spacy
    print("\033[1mFrequency Distribution for spacy\033[0m")

    print(milton_freq_list.get('ALL', 0))
    print(milton_freq_list.get('All', 0))
    print(milton_freq_list.get('all', 0))

    lower_bound = float(96) # Change these two number to compute the required cut offs
    upper_bound = float(604)
    lexicon_cut_off = len(cut_off(milton_freq_list, n_min=lower_bound, n_max=upper_bound))

    print('Original', len(milton_freq_list))
    print('CutOFF Min:', lower_bound, 'MAX:', upper_bound, ' Lexicon Size:', lexicon_cut_off)

    best_words = nbest(milton_freq_list,n=5)
    print(best_words)

    #Frequency Distribution for nltk
    print("\033[1mFrequency Distribution for nltk\033[0m")

    print(milton_freq_list.get('ALL', 0))
    print(milton_freq_list.get('All', 0))
    print(milton_freq_list.get('all', 0))

    lower_bound = float(96) # Change these two number to compute the required cut offs
    upper_bound = float(604)
    lexicon_cut_off = len(cut_off(milton_freq_list, n_min=lower_bound, n_max=upper_bound))

    print('Original', len(milton_freq_list))
    print('CutOFF Min:', lower_bound, 'MAX:', upper_bound, ' Lexicon Size:', lexicon_cut_off)

    best_words = nbest(milton_freq_list,n=5)
    print(best_words)
    
if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    main()