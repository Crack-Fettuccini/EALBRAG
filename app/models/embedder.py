import torch
import torch.nn as nn

class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weights=None, freeze=False, dropout=0.1):
        super(TokenEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if pretrained_weights is not None:
            self.embedding.weight = nn.Parameter(pretrained_weights)
        
        if freeze:
            self.embedding.weight.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        return self.dropout(embeddings)

