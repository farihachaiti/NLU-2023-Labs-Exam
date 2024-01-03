# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import torch.optim as optim

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    



    lr = 0.0001 # learning rate
    e = 1e-08

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(vocab_len, out_int, out_slot)

    optimizer = Adam(learning_rate=lr, epsilon=e)
    
    losses = [SparseCategoricalCrossentropy(from_logits=False),
            SparseCategoricalCrossentropy(from_logits=False)]
    metrics = [SparseCategoricalAccuracy(name='slot_accuracy'), SparseCategoricalAccuracy(name='intent_accuracy')]

    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    train_input_ids, train_attention_masks, train_token_type_ids, train_labels, train_slots = prepare_dataset(train_raw, model)
    dev_input_ids, dev_attention_masks, dev_token_type_ids, dev_labels, dev_slots = prepare_dataset(dev_raw, model)
    test_input_ids, test_attention_masks, test_token_type_ids, test_labels, test_slots = prepare_dataset(test_raw, model)



    history = model.fit(
        [train_input_ids,train_attention_masks], (train_slots, train_labels),
        validation_data=([dev_input_ids,dev_attention_masks], (dev_slots, dev_labels)),
        epochs=15, batch_size=128)


    result = model.evaluate([test_input_ids,test_attention_masks], (test_slots, test_labels))
    # Save the model to the bin file
    model.save("bin/saved_model")
    print("\033[1mResults of BERT MODEL:\033[0m")
    print(f'Loss: {result[0]}')
    print(f'Slot Loss: {result[1]}')
    print(f'Intent Loss: {result[2]}')
    print(f'Slot Accuracy: {result[3]}')
    print(f'Intent Accuracy: {result[4]}')