import numpy as np
import torch
from dataset import ScoreDataset
from model import Transformer
from config import CFG

cfg = CFG()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=cfg.NUM_TOKENS, dim_model=cfg.dim_model, num_heads=cfg.num_heads, 
    num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers, dropout_p=0.1
).to(device)
checkpoint = torch.load(cfg.save_checkpoint_dir)
model.load_state_dict(checkpoint['model_state_dict'])


def predict(model, input_sequence, max_length=2048, PAD=0, EOS=1):

    model.eval()
    y_input = torch.tensor([[PAD]], dtype=torch.long, device=device)

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS:
            break

    return y_input.view(-1).tolist()
  
  
def print_sequence(seq, EOS):
    for token in seq:
        print(token, end='')
        print(' ', end='')
        if token ==  EOS:
            print()
            return 

valset = ScoreDataset(cfg.input_val_dir, cfg.target_val_dir, [cfg.PAD, cfg.EOS, cfg.MAXLEN])
idx = np.random.randint(0, len(valset) - 1)
print(f"Example index: {idx}")

sample, gt = valset[idx]
print("Input")
print_sequence(sample, cfg.EOS)

sample = torch.tensor(sample, dtype=torch.long, device=device).unsqueeze(0)
result = predict(model, sample, max_length=cfg.MAXLEN, PAD=cfg.PAD, EOS=cfg.EOS)
print("Prediction")
print_sequence(result, cfg.EOS)

print("Ground-truth")
print_sequence(gt, cfg.EOS)

