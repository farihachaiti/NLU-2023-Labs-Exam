# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from pprint import pprint
import nltk
import spacy
from spacy_conll import init_parser
nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank
from spacy.tokenizer import Tokenizer

import stanza
import spacy_stanza

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    print("\033[1mParsing and Evaluating using Spacy\033[0m")

    # Load the spacy model
    nlp = spacy.load("en_core_web_sm")

    nlp.tokenizer = Tokenizer(nlp.vocab)


    # Set up the conll formatter 
    config = {"ext_names": {"conll_pd": "pandas"},
            "conversion_maps": {"deprel": {"nsubj": "subj"}}}

    # Add the formatter to the pipeline
    nlp.add_pipe("conll_formatter", config=config, last=True)
    nlp.tokenizer = Tokenizer(nlp.vocab)

    parse_and_evaluate_sents(dependency_treebank, nlp)


    print("\033[1mParsing and Evaluating using Stanza\033[0m")
    nlp = init_parser("nl", "stanza", parser_opts={"verbose": False})
    nlp = spacy_stanza.load_pipeline("en", verbose=False, tokenize_pretokenized=True)

    config = {"ext_names": {"conll_pd": "pandas"}, "conversion_maps": {"DEPREL": {"nsubj": "subj", "root" : "ROOT"}}}

    # Add the formatter to the pipeline
    nlp.add_pipe("conll_formatter", config=config, last=True)
    
    parse_and_evaluate_sents(dependency_treebank, nlp)

    print("\033[1mAs we can see from the output, the dependency tags for Spacy and Stanza are not same\033[0m")