import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import ScoreDataset
from model import Transformer
from config import CFG
import argparse
import time


cfg = CFG()

def parse_args():
    parser = argparse.ArgumentParser(description="Lyrics Alignment Transformer")
    parser.add_argument(
        "--ckpt_dir", type=str, default=None, help="directory of checkpoint file"
    )
    parser.add_argument(
        "--epoch", type=int, default=5, help="number of epochs, with default 5"
    )
    args = parser.parse_args()
    return args


def train_loop(model, opt, loss_fn, dataloader):

    model.train()
    total_loss = 0
    
    datanum = 0
    current_cut = cfg.log_period
    start_time = time.time()

    for batch in dataloader:
        X, y = batch
        X, y = X.to(device), y.to(device)

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

        datanum += cfg.BATCH_SIZE
        if datanum > current_cut:
            curr_time = time.time() - start_time
            print("\t{:4d}/{:4d} samples trained. Elapsed time: {:6.2f} s".format(current_cut, cfg.BATCH_SIZE * len(dataloader), curr_time))
            current_cut += cfg.log_period

    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):

    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:      
            X, y = batch     
            X, y = X.to(device), y.to(device)

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


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, curr_epoch, loss_list):

    print("Start of Training")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {curr_epoch + epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)      # 1 epoch
        loss_list[0].append(train_loss)
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        loss_list[1].append(validation_loss)
        
        torch.save({
            'epoch': curr_epoch + epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss_list,
            }, cfg.save_checkpoint_dir)
        
        print("Checkpoint saved")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()
    
    print("End of Training")
        
    return loss_list


args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=cfg.NUM_TOKENS, dim_model=cfg.dim_model, num_heads=cfg.num_heads, 
    num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers, dropout_p=0.1
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()    

# Load checkpoint
if args.ckpt_dir:
    checkpoint = torch.load(args.ckpt_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    curr_epoch = checkpoint['epoch']
    loss_list = checkpoint['loss']
else:
    curr_epoch = 0
    loss_list = [[], []]    
    
trainset = ScoreDataset(cfg.input_train_dir, cfg.target_train_dir, [cfg.PAD, cfg.EOS, cfg.MAXLEN])
valset = ScoreDataset(cfg.input_val_dir, cfg.target_val_dir, [cfg.PAD, cfg.EOS, cfg.MAXLEN])

train_dataloader = DataLoader(trainset, batch_size=cfg.BATCH_SIZE, shuffle=False)
val_dataloader = DataLoader(valset, batch_size=cfg.BATCH_SIZE, shuffle=False)


if __name__ == '__main__':
    loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, args.epoch, curr_epoch, loss_list)
    epoch_list = [i + 1 for i in range(len(loss_list[0]))]

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epoch_list, loss_list[0], 'b', label='train')
    plt.plot(epoch_list, loss_list[1], 'r', label='val')
    plt.legend(loc='upper right')
    plt.savefig("loss_graph.jpg", dpi=300)    # Save graph as a JPG image
