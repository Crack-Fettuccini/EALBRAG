# models/lstm_memory.py
import torch
import torch.nn as nn

class LSTMMemory(nn.Module):
    def __init__(self, embed_dim, hidden_dim=512, num_layers=1):
        """
        Initialize the LSTM Memory module.
        :param embed_dim: Dimensionality of the embeddings.
        :param hidden_dim: Number of features in the hidden state of the LSTM.
        :param num_layers: Number of recurrent layers in the LSTM.
        """
        super(LSTMMemory, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Define linear layers to project LSTM outputs back to embed_dim
        self.proj = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the LSTM Memory.
        :param x: Input tensor of shape (batch, seq_len, embed_dim)
        :param hidden: Tuple of (h_0, c_0) for LSTM
        :return: Projected memory tensor of shape (batch, seq_len, embed_dim), updated hidden states
        """
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (batch, seq_len, hidden_dim)
        proj_out = self.proj(lstm_out)  # proj_out: (batch, seq_len, embed_dim)
        return proj_out, hidden
