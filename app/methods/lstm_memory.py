import torch
import torch.nn as nn

class LSTMMemory(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 512, num_layers: int = 1):
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

        # Define the LSTM with batch_first = True for better handling of batches
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Linear projection layer to project LSTM output back to embed_dim
        self.proj = nn.Linear(hidden_dim, embed_dim)

        # Initialize hidden and cell states
        self.hidden = None

    def reset_hidden(self, batch_size: int = 1):
        """
        Resets the LSTM hidden state.
        :param batch_size: The batch size for which to reset hidden states.
        """
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        )

    def forward(self, x: torch.Tensor, hidden: tuple = None):
        """
        Forward pass through the LSTM Memory.
        :param x: Input tensor of shape (batch, seq_len, embed_dim).
        :param hidden: Optional hidden state for LSTM. If None, use internal hidden state.
        :return: Projected memory tensor of shape (batch, seq_len, embed_dim), updated hidden states.
        """
        batch_size = x.size(0)
        
        # Use existing hidden state if not provided
        if hidden is None:
            if self.hidden is None or self.hidden[0].size(1) != batch_size:
                self.reset_hidden(batch_size)
            hidden = self.hidden
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (batch, seq_len, hidden_dim)

        # Project the LSTM output to embedding dimension
        proj_out = self.proj(lstm_out)  # proj_out: (batch, seq_len, embed_dim)

        # Update internal hidden state
        self.hidden = hidden

        return proj_out, hidden
