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

    #Descriptive statistics on the reference sentences and tokens
    print("\033[1mDescriptive statistics on the reference sentences and tokens\033[0m")
    statistics(milton_chars, milton_words, milton_sents)
    

    # process the document with spacy
    doc = nlp(milton_chars)
    tokens = [token.text for token in doc if token.is_alpha]

    #Descriptive statistics on the automatically processed corpus by spacy
    print("\033[1mDescriptive statistics on the automatically processed corpus by spacy\033[0m")
    statistics(milton_chars, tokens, format(list(doc.sents)))


    #Descriptive statistics on the automatically processed corpus by nltk
    print("\033[1mDescriptive statistics on the automatically processed corpus by nltk\033[0m")
    statistics(milton_chars, milton_words_nltk, milton_sents_nltk)


    #Lowercased lexicons for reference
    milton_lexicon = set([w.lower() for w in milton_words])

    print("\033[1mLowercased lexicons for reference\033[0m")
    print_lowercased_lexicons(milton_lexicon)

    #Lowercased lexicons for spacy
    milton_lexicon_spacy = set([w.lower() for w in tokens])

    print("\033[1mLowercased lexicons for spacy\033[0m")
    print_lowercased_lexicons(milton_lexicon_spacy)

    #Lowercased lexicons for nltk
    milton_lexicon_nltk = set([w.lower() for w in milton_words_nltk])

    print("\033[1mLowercased lexicons for nltk\033[0m")
    print_lowercased_lexicons(milton_lexicon_nltk)

    #Frequency Distribution for reference
    print("\033[1mFrequency Distribution for reference\033[0m")
    lower_bound = float(96) # Change these two number to compute the required cut offs
    upper_bound = float(604)
    milton_freq_list = Counter(milton_words)
    frequency_distribution(milton_freq_list, lower_bound, upper_bound)

    #Frequency Distribution for spacy
    print("\033[1mFrequency Distribution for spacy\033[0m")
    lower_bound = float(96) # Change these two number to compute the required cut offs
    upper_bound = float(604)
    milton_freq_list = Counter(tokens)
    frequency_distribution(milton_freq_list, lower_bound, upper_bound)

    #Frequency Distribution for nltk
    print("\033[1mFrequency Distribution for nltk\033[0m")
    lower_bound = float(96) # Change these two number to compute the required cut offs
    upper_bound = float(604)
    milton_freq_list = Counter(milton_words_nltk)
    frequency_distribution(milton_freq_list, lower_bound, upper_bound)
    
if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    main()