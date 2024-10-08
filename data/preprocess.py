import torch

def load_shakespeare_data(seq_len):
    text = open("data/tiny_shakespeare.txt").read()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    token_ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    
    sequences = [token_ids[i:i+seq_len] for i in range(0, len(token_ids) - seq_len, seq_len)]
    
    return sequences, stoi, itos
