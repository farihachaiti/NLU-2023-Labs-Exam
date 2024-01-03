# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

#function to print pcfg rules for the chosen two sentences
def PCFG_rule():
        #PCFG
    wght_rules = [
        'S -> NP VP PP CONJP [1.0]',
        'NP -> N VP [0.5]',
        'NP -> PRON VP [0.5]',
        'VP -> V PP [0.7]',
        'VP -> V N [0.3]',
        'PP -> P DET N [0.3]',
        'PP -> P N [0.2]',
        'PP -> P VP [0.5]',
        'CONJP -> CC NP VP [0.6]',
        'CONJP -> CC N [0.4]',
        'PRON -> "I" [0.5]',
        'PRON -> "She" [0.5]', 
        'P -> "with" [1.0]',
        'N -> "Margaret" [0.5]',
        'N -> "rice" [0.1]',
        'N -> "fish" [0.1]', 
        'N -> "beef" [0.1]',
        'N -> "dance" [0.1]',
        'N -> "photography" [0.1]',
        'V -> "like" [0.25]',
        'V -> "do" [0.25]',
        'V -> "ate" [0.25]',
        'V -> "loves" [0.25]',
        'DET -> "a" [1.0]',
        'CC -> "and" [0.5]',
        'CC -> "but" [0.5]',
        'ADV -> "also" [0.5]',
        'ADV -> "not" [0.5]',
    ]

    return wght_rules

#function to parse sentences
def parse_sentences(sentences, parser):
    for sent in sentences:
        for tree in parser.parse(sent.split()):
            print(tree)

#function to print sentences
def print_sentences(func, grammer=None, start=None, n=10):
    if grammer:
        for sent in func(grammer, start=start, n=n):
            print(sent)
    else:
        for sent in func(n):
            print(sent)
