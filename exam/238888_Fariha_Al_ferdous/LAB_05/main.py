# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import nltk
from nltk.parse.generate import generate
from nltk import Nonterminal
from pcfg import PCFG



if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
        #2 sentences
    sentences = ["Margaret ate rice with a fish but I do not like fish", "She loves photography and also dance"]


    print("\033[1m2 Sentences:\033[0m")
    for sent in sentences:   
        print(sent)

        
    print("\033[1mPCFG Model:\033[0m")
    wght_rules = PCFG_rule()
    for rule in wght_rules:
        print(rule)

    grammer = nltk.PCFG.fromstring(wght_rules)
    parser = nltk.ViterbiParser(grammer)


    print("\033[1mParsing of Sentences:\033[0m")
    parse_sentences(sentences, parser)



    print("\033[1m10 sentences with nltk.parse.generate.generate using starting symbole 'NP'\033[0m") 
    start = Nonterminal('NP')
    print_sentences(generate, grammer=grammer, start=start, n=10)

    print("\033[1m10 sentences with nltk.parse.generate.generate using starting symbol 'VP'\033[0m") 
    start = Nonterminal('VP')
    print_sentences(generate, grammer=grammer, start=start, n=10)

    #10 sentences with PCFG.generate    
    print("\033[1m10 sentences with PCFG.generate\033[0m") 
    toy_grammar = PCFG.fromstring(wght_rules)
    print_sentences(toy_grammar.generate, grammer=None, start=None, n=10)
