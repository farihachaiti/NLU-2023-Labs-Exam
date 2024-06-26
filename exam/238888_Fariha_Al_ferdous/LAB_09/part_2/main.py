# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    
# Experiment also with a smaller or bigger model by changing hid and emb sizes 
# A large model tends to overfit
    hid_size = 200
    emb_size = 200

    #train and test regularized model
    # With SGD try with an higer learning rate
    lr = 0.01 # This is definitely not good for SGD
    clip = 3 # Clip the gradient
    device = 'cuda:0'

    vocab_len = len(lang.word2id)

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], tie_weights=True).to(device)
    model.apply(init_weights)


    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')




    n_epochs = 100
    patience = 50
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = model
    best_loss = []
    pbar = tqdm(range(1,n_epochs+1))

    # Initialize both optimizers
    optimizer_asgd = optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=1.2e-6)
    optimizer_adamw = optim.AdamW(model.parameters(), lr=lr)
    optimizer = optimizer_adamw
    for epoch in pbar:
        #Applying non-monotonic ASGD taking threshold for window size of 10:
        if epoch % 1 == 0:
            if len(best_loss)>= 10 and np.mean(best_loss[-10:])>1:
                optimizer = optimizer_asgd
        else:
            optimizer = optimizer_adamw
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        best_loss.append(loss)
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss).mean())
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        losses_dev.append(np.asarray(loss_dev).mean())
        
        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience += 50
        else:
            patience -= 1
        pbar.set_description("PPL: %f" % best_ppl)
        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean            
    best_model.to(device)
    # Specify the path to save the bin file
    bin_file_path = "bin/model.bin"

    # Save the model to the bin file
    torch.save(best_model.state_dict(), bin_file_path)
    #results
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)   
    print("\033[1mPPL using LSTM with variational dropout, tied weights and non-monotonic ASGD:\033[0m")
    print('Test ppl: ', final_ppl)