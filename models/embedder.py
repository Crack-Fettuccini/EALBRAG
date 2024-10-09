import torch
import torch.nn as nn

class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, tokens):
        return self.embedding(tokens)
