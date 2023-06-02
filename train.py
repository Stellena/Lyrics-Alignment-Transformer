import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from dataset import ScoreDataset
from model import Transformer


def train_loop(model, opt, loss_fn, dataloader):

    model.train()
    total_loss = 0
    
    datanum = 0
    current_cut = 500

    for batch in dataloader:
        X, y = batch

        flag = False
        for sample in X:
            for token in sample:
                assert token >= 0

        #X, y = batch[:, 0], batch[:, 1]     # 각각 src, tgt. Shape: (batch_size, max_seq_len)
        #pad = np.zeros((MAXLEN - X.shape[0]), dtype=X.dtype)
        #X = np.concatenate((X, pad)).reshape((1, -1))
        #pad = np.zeros((MAXLEN - y.shape[0]), dtype=y.dtype)
        #y = np.concatenate((y, pad)).reshape((1, -1))

        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:,:-1]
        y_expected = y[:,1:]
        
        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)      
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()

        datanum += BATCH_SIZE
        if datanum > current_cut:
            print("\t{} samples trained".format(current_cut))
            current_cut += 500

    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:      
            X, y = batch
            #pad = np.zeros((MAXLEN - X.shape[0]), dtype=X.dtype)
            #X = np.concatenate((X, pad)).unsqueeze(0)
            #pad = np.zeros((MAXLEN - y.shape[0]), dtype=y.dtype)
            #y = np.concatenate((y, pad)).unsqueeze(0)            
            
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            y_expected = y[:,1:]
            
            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            pred = model(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)      
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)      # 1 epoch
        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list


PAD = 0
EOS = 1
MAXLEN = 2048
BATCH_SIZE = 8          # 너무 크게 잡으면 램용량 초과 뜰 수 있음.
NUM_TOKENS = 755        # 토큰 종류 수
EPOCHS = 5


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=NUM_TOKENS, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()    


trainset = ScoreDataset("input_train_lyrics.npy", "target_train_lyrics.npy", [PAD, EOS, MAXLEN])
valset = ScoreDataset("input_val_lyrics.npy", "target_val_lyrics.npy", [PAD, EOS, MAXLEN])

train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)


train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, EPOCHS)
