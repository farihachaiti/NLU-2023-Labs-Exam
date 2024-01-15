# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import nltk
import spacy
import en_core_web_sm
from collections import Counter

nltk.download('gutenberg')



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
    tokens = [token.text for token in doc]

    #Descriptive statistics on the automatically processed corpus by spacy
    print("\033[1mDescriptive statistics on the automatically processed corpus by spacy\033[0m")
    statistics(milton_chars, tokens, format(list(doc.sents)))


    #Descriptive statistics on the automatically processed corpus by nltk
    print("\033[1mDescriptive statistics on the automatically processed corpus by nltk\033[0m")
    statistics(milton_chars, milton_words_nltk, milton_sents_nltk)


    #Lowercased lexicons for reference

    print("\033[1mLexicon sizes for the Lowercased lexicons for reference\033[0m")
    print_lowercased_lexicons(milton_words)

    #Lowercased lexicons for spacy
    print("\033[1mLexicon sizes for the Lowercased lexicons for spacy\033[0m")
    print_lowercased_lexicons(tokens)

    #Lowercased lexicons for nltk
    print("\033[1mLexicon sizes for the Lowercased lexicons for nltk\033[0m")
    print_lowercased_lexicons(milton_words_nltk)

    #Frequency Distribution for reference
    print("\033[1mTop 5 frequencies for Frequency Distribution for reference\033[0m")
    frequency_distribution_ref(milton_words)

    #Frequency Distribution for spacy
    print("\033[1mTop 5 frequencies for Frequency Distribution for spacy\033[0m")
    frequency_distribution_nltk(tokens)

    #Frequency Distribution for nltk
    print("\033[1mTop 5 frequencies for Frequency Distribution for nltk\033[0m")
    frequency_distribution_spacy(milton_words_nltk)



if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    main()