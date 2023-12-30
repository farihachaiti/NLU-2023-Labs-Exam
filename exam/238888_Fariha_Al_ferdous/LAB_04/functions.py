# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
def nlp_accuracy(reference, test):
    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    return sum(x == y for x, y in zip(reference, test)) / len(test)