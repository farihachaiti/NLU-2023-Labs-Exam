# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from nltk.parse.dependencygraph import DependencyGraph
from nltk.parse import DependencyEvaluator

def parse_and_evaluate_sents(dependency_treebank, nlp):
    docs = []
    dps = []

    #last 100 sentences
    for sent in dependency_treebank.sents()[-100:]:
        # Parse the sentence
        docs.append(nlp(' '.join(sent)))

    # Convert doc to a pandas object
    for doc in docs:
        df = doc._.pandas

    # Select the columns accoroding to Malt-Tab format
        tmp = df[["FORM",'XPOS','HEAD','DEPREL']].to_string(header=False, index=False)

    # See the outcome
        print(tmp)
        
    # Get finally our the DepencecyGraph
        dp = DependencyGraph(tmp)
        print('\033[1mTree:\033[0m')
        dp.tree().pretty_print(unicodelines=True, nodedist=4)
        dps.append(dp)

    de = DependencyEvaluator(dps, dependency_treebank.parsed_sents()[-100:])
    las, uas = de.eval()
    # no labels, thus identical
    print("\033[1mLAS\033[0m")
    print("{:.3}".format(las))
    print("\033[1mUAS\033[0m")
    print("{:.3}".format(uas))