# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
import torch
torch.cuda.empty_cache()
torch.version.cuda
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os 
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.is_available()
device = 'cuda:0'

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    
    #train and test models with variations


    print("\033[1mPPL using LSTM instead of RNN with SGD:\033[0m")
    hid_size = 200
    emb_size = 300

    # Don't forget to experiment with a lower training batch size

    # With SGD try with an higer learning rate
    lr = 0.0001 # This is definitely not good for SGD
    clip = 5 # Clip the gradient


    vocab_len = len(lang.word2id)

    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)
    # Specify the path to save the bin file

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = model
    pbar = tqdm(range(1,n_epochs))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            pbar.set_description("PPL: %f" % best_ppl)
            if patience <= 0: # Early stopping with patience
                
                break # Not nice but it keeps the code clean
    #saving model                
    best_model.to(device)
    bin_file_path = "bin/model.bin"

    # Save the model to the bin file
    torch.save(model.state_dict(), bin_file_path)
    #results
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)


    print("\033[1mPPL using LSTM with two dropout layers and SGD:\033[0m")
 
    # Experiment also with a smaller or bigger model by changing hid and emb sizes 
    # A large model tends to overfit
    hid_size = 200
    emb_size = 300

    # Don't forget to experiment with a lower training batch size

    # With SGD try with an higer learning rate
    lr = 0.01 # This is definitely not good for SGD
    clip = 5 # Clip the gradient

    vocab_len = len(lang.word2id)

    model = LM_LSTM_with_dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')


    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = model
    pbar = tqdm(range(1,n_epochs))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            pbar.set_description("PPL: %f" % best_ppl)
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean
                            
    best_model.to(device)
    # Specify the path to save the bin file
    bin_file_path = "bin/model_with_dropout.bin"

    # Save the model to the bin file
    torch.save(model.state_dict(), bin_file_path)
    #results
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    

    print('Test ppl: ', final_ppl)

    print("\033[1mPPL using LSTM with two dropout layers and AdamW:\033[0m")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001) #replaced SGD with AdamW
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = model
    pbar = tqdm(range(1,n_epochs))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            pbar.set_description("PPL: %f" % best_ppl)
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean
                            
    best_model.to(device)
    # Specify the path to save the bin file
    bin_file_path = "bin/model_with_dropout.bin"

    # Save the model to the bin file
    torch.save(best_model.state_dict(), bin_file_path)
    #results
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    

    print('Test ppl: ', final_ppl)