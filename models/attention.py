# models/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    def __init__(self, embed_dim, max_memory_size=1000, lstm_hidden_size=256, num_layers=1):
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
        keys = keys.view(-1, self.embed_dim)  # Flatten
        values = values.view(-1, self.embed_dim)
        attention_scores = attention_scores.view(-1)

        mask = attention_scores > threshold
        selected_keys = keys[mask]
        selected_values = values[mask]

        available_space = self.max_memory_size - self.current_size
        if available_space <= 0:
            return

        if selected_keys.size(0) > available_space:
            selected_keys = selected_keys[:available_space]
            selected_values = selected_values[:available_space]

        if selected_keys.size(0) > 0:
            lstm_input = selected_keys.unsqueeze(1)
            lstm_out, _ = self.lstm(lstm_input)
            self.keys[self.current_size:self.current_size + lstm_out.size(1)] = lstm_out[:, -1, :]
            self.values[self.current_size:self.current_size + lstm_out.size(1)] = selected_values[:lstm_out.size(1)]
            self.current_size += lstm_out.size(1)

    def retrieve_memory(self, query_embeddings, top_k=5):
        if self.current_size == 0:
            return None

        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        memory_keys = F.normalize(self.keys[:self.current_size], p=2, dim=1)

        similarities = torch.matmul(query_norm, memory_keys.T)
        topk_values, topk_indices = torch.topk(similarities, top_k, dim=-1, largest=True, sorted=True)

        retrieved_values = self.values[topk_indices]

        return retrieved_values

    def reset_memory(self):
        self.current_size = 0
        self.keys = torch.zeros((self.max_memory_size, self.embed_dim)).to(torch.float32)
        self.values = torch.zeros((self.max_memory_size, self.embed_dim)).to(torch.float32)

class AttentionWithMemory(nn.Module):
    def __init__(self, embed_dim, num_heads, max_memory_size=1000, lstm_hidden_size=256, num_layers=1):
        super(AttentionWithMemory, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.memory_bank = MemoryBank(embed_dim, max_memory_size, lstm_hidden_size, num_layers)
        
        self.attention_head_size = embed_dim // num_heads
        self.all_head_size = self.attention_head_size * num_heads
        
        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()

        # Compute Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = mixed_query_layer.view(batch_size, seq_length, self.num_heads, self.attention_head_size).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_size)
        key_layer = mixed_key_layer.view(batch_size, seq_length, self.num_heads, self.attention_head_size).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_size)
        value_layer = mixed_value_layer.view(batch_size, seq_length, self.num_heads, self.attention_head_size).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_size)

        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / torch.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Store attention scores and update memory
        self.memory_bank.update_memory(key_layer, value_layer, attention_probs)

        # Retrieve memory based on current hidden states
        memory_values = self.memory_bank.retrieve_memory(hidden_states)

        # Weighted average of values and memory
        if memory_values is not None:
            value_layer = torch.cat((value_layer, memory_values), dim=2)  # (batch_size, num_heads, seq_length, head_size + retrieved)
            attention_probs = F.softmax(attention_scores, dim=-1)  # Recalculate probs after retrieving memory

        # Context layer
        context_layer = torch.matmul(attention_probs, value_layer)  # (batch_size, num_heads, seq_length, head_size)
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_length, self.all_head_size)  # (batch_size, seq_length, all_head_size)

        output_layer = self.output(context_layer)  # (batch_size, seq_length, embed_dim)

        return output_layer
