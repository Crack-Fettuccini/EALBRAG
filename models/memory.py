import torch
import torch.nn as nn
import torch.nn.functional as F
from lstm_memory import LSTMMemory

class MemoryBank(nn.Module):
    def __init__(self, embed_dim: int, max_memory_size: int = 1000, lstm_hidden_size: int = 512, num_layers: int = 1):
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
        self.current_size = 0  # Tracks current memory size
        
        # Initialize LSTM Memory for long-term memory encoding
        self.lstm_memory = LSTMMemory(embed_dim, hidden_dim=lstm_hidden_size, num_layers=num_layers)
        
        # Memory storage (circular buffer)
        self.keys = torch.zeros((max_memory_size, embed_dim), dtype=torch.float32)
        self.values = torch.zeros((max_memory_size, embed_dim), dtype=torch.float32)
        self.write_index = 0  # Tracks position for new memory entries

    def update_memory(self, keys: torch.Tensor, values: torch.Tensor, attention_scores: torch.Tensor, threshold: float = 0.5):
        """
        Updates the memory bank based on attention scores.
        :param keys: Tensor of shape (batch, seq_len, embed_dim).
        :param values: Tensor of shape (batch, seq_len, embed_dim).
        :param attention_scores: Tensor of shape (batch, seq_len), used to select which key-value pairs to store.
        :param threshold: Threshold to determine which keys to store based on attention scores.
        """
        keys = keys.view(-1, self.embed_dim)  # Flatten (batch*seq_len, embed_dim)
        values = values.view(-1, self.embed_dim)  # Flatten (batch*seq_len, embed_dim)
        attention_scores = attention_scores.view(-1)  # Flatten (batch*seq_len)

        # Select keys and values above the threshold
        mask = attention_scores > threshold
        selected_keys = keys[mask]  # (num_selected, embed_dim)
        selected_values = values[mask]  # (num_selected, embed_dim)

        num_selected = selected_keys.size(0)
        if num_selected == 0:
            return  # No valid memory to update

        # Pass selected keys through LSTM Memory
        lstm_input = selected_keys.unsqueeze(0)  # Add batch dimension for LSTM
        memory_outputs, _ = self.lstm_memory(lstm_input)  # Output shape: (1, num_selected, embed_dim)
        memory_outputs = memory_outputs.squeeze(0)  # (num_selected, embed_dim)

        # Eviction or writing new memory entries
        for i in range(num_selected):
            # Write memory in a circular manner
            self.keys[self.write_index] = memory_outputs[i]
            self.values[self.write_index] = selected_values[i]
            self.write_index = (self.write_index + 1) % self.max_memory_size  # Circular buffer write
            
            # Update current memory size, capped at max_memory_size
            if self.current_size < self.max_memory_size:
                self.current_size += 1

    def retrieve_memory(self, query_embeddings: torch.Tensor, top_k: int = 5):
        """
        Retrieves top-k relevant memory entries based on query embeddings.
        :param query_embeddings: Tensor of shape (batch, embed_dim).
        :param top_k: Number of memory entries to retrieve per query.
        :return: Retrieved values of shape (batch, top_k, embed_dim).
        """
        if self.current_size == 0:
            return None  # No memory to retrieve from

        # Normalize embeddings for cosine similarity
        query_norm = F.normalize(query_embeddings, p=2, dim=1)  # (batch, embed_dim)
        memory_keys_norm = F.normalize(self.keys[:self.current_size], p=2, dim=1)  # (current_size, embed_dim)

        # Compute cosine similarity
        similarities = torch.matmul(query_norm, memory_keys_norm.T)  # (batch, current_size)

        # Get top-k indices and corresponding values
        topk_values, topk_indices = torch.topk(similarities, top_k, dim=-1)  # (batch, top_k)
        retrieved_values = self.values[topk_indices]  # (batch, top_k, embed_dim)

        return retrieved_values

    def reset_memory(self):
        """
        Resets the memory bank and LSTM state.
        """
        self.current_size = 0
        self.write_index = 0
        self.keys.fill_(0)
        self.values.fill_(0)
        self.lstm_memory.reset_hidden(batch_size=1)  # Reset LSTM hidden state
