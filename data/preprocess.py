# data/preprocess.py
import torch
from torch.utils.data import Dataset

def load_shakespeare_data(seq_len=32):
    """
    Load and preprocess the Tiny Shakespeare dataset.
    :param seq_len: Sequence length for training
    :return: sequences list, stoi dict, itos dict
    """
    # Load the dataset from file
    with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Create character-level tokenizer
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Convert text to token IDs
    token_ids = [stoi[ch] for ch in text]

    # Create sequences
    sequences = []
    for i in range(len(token_ids) - seq_len):
        seq_in = token_ids[i:i + seq_len]
        seq_out = token_ids[i + 1:i + 1 + seq_len]
        sequences.append((seq_in, seq_out))

    return sequences, stoi, itos

class ShakespeareDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_in, seq_out = self.sequences[idx]
        return torch.tensor(seq_in, dtype=torch.long), torch.tensor(seq_out, dtype=torch.long)
