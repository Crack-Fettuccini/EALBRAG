# models/memory.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    def __init__(self, embed_dim, max_memory_size=1000, lstm_hidden_size=256, num_layers=1):
        """
        Initializes the MemoryBank with LSTM for long-term memory.
        :param embed_dim: Dimensionality of the embeddings.
        :param max_memory_size: Maximum number of key-value pairs to store.
        :param lstm_hidden_size: Number of features in the hidden state of the LSTM.
        :param num_layers: Number of recurrent layers in the LSTM.
        """
        super(MemoryBank, self).__init__()
        self.embed_dim = embed_dim
        self.max_memory_size = max_memory_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        
        # LSTM for memory handling
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_size, num_layers=num_layers, batch_first=True)

        # Initialize keys and values
        self.keys = torch.zeros((max_memory_size, embed_dim)).to(torch.float32)
        self.values = torch.zeros((max_memory_size, embed_dim)).to(torch.float32)
        self.current_size = 0

    def update_memory(self, keys, values, attention_scores, threshold=0.5):
        """
        Updates the memory bank based on attention scores and stores in LSTM.
        :param keys: Tensor of shape (batch, seq_len, embed_dim)
        :param values: Tensor of shape (batch, seq_len, embed_dim)
        :param attention_scores: Tensor of shape (batch, seq_len)
        :param threshold: Threshold to determine which keys to store.
        """
        # Flatten batch and sequence dimensions
        keys = keys.view(-1, self.embed_dim)  # (batch*seq_len, embed_dim)
        values = values.view(-1, self.embed_dim)  # (batch*seq_len, embed_dim)
        attention_scores = attention_scores.view(-1)  # (batch*seq_len)

        # Select keys and values above the threshold
        mask = attention_scores > threshold
        selected_keys = keys[mask]  # (num_selected, embed_dim)
        selected_values = values[mask]  # (num_selected, embed_dim)

        # Determine available space in memory
        available_space = self.max_memory_size - self.current_size
        if available_space <= 0:
            return  # Memory is full; could implement eviction policy here

        # Truncate if necessary
        if selected_keys.size(0) > available_space:
            selected_keys = selected_keys[:available_space]
            selected_values = selected_values[:available_space]

        # Store selected keys and values in LSTM
        if selected_keys.size(0) > 0:
            lstm_input = selected_keys.unsqueeze(1)  # Add sequence dimension
            lstm_out, _ = self.lstm(lstm_input)
            self.keys[self.current_size:self.current_size + lstm_out.size(1)] = lstm_out[:, -1, :]  # Store last LSTM output
            self.values[self.current_size:self.current_size + lstm_out.size(1)] = selected_values[:lstm_out.size(1)]
            self.current_size += lstm_out.size(1)

    def retrieve_memory(self, query_embeddings, top_k=5):
        """
        Retrieves top-k relevant memory entries based on query embeddings.
        :param query_embeddings: Tensor of shape (batch, embed_dim)
        :param top_k: Number of memory entries to retrieve per query.
        :return: Retrieved values of shape (batch, top_k, embed_dim)
        """
        if self.current_size == 0:
            return None  # No memory to retrieve from

        # Normalize embeddings
        query_norm = F.normalize(query_embeddings, p=2, dim=1)  # (batch, embed_dim)
        memory_keys = F.normalize(self.keys[:self.current_size], p=2, dim=1)  # (current_size, embed_dim)

        # Compute cosine similarity
        similarities = torch.matmul(query_norm, memory_keys.T)  # (batch, current_size)

        # Get top-k indices
        topk_values, topk_indices = torch.topk(similarities, top_k, dim=-1, largest=True, sorted=True)  # (batch, top_k)

        # Gather the corresponding values
        retrieved_values = self.values[topk_indices]  # (batch, top_k, embed_dim)

        return retrieved_values  # (batch, top_k, embed_dim)

    def reset_memory(self):
        """
        Resets the memory bank and LSTM state.
        """
        self.current_size = 0
        self.keys = torch.zeros((self.max_memory_size, self.embed_dim)).to(torch.float32)
        self.values = torch.zeros((self.max_memory_size, self.embed_dim)).to(torch.float32)
