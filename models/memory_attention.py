# models/memory_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.memory import MemoryBank

class MemoryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, memory_size=1000, lstm_hidden_size=256, num_lstm_layers=1):
        """
        Initializes the MemoryAttention module.
        :param embed_dim: Dimensionality of the embeddings.
        :param num_heads: Number of attention heads.
        :param memory_size: Maximum number of memory slots.
        :param lstm_hidden_size: Number of features in the hidden state of the LSTM.
        :param num_lstm_layers: Number of recurrent layers in the LSTM.
        """
        super(MemoryAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        
        # Projection layers
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scaled dot-product attention
        self.scale = self.head_dim ** -0.5
        
        # Memory Bank with LSTM
        self.memory_bank = MemoryBank(
            embed_dim=embed_dim,
            max_memory_size=memory_size,
            lstm_hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers
        )
        
    def forward(self, query, key, value, attention_mask=None):
        """
        Forward pass for the MemoryAttention mechanism.
        :param query: Tensor of shape (batch, query_len, embed_dim)
        :param key: Tensor of shape (batch, key_len, embed_dim)
        :param value: Tensor of shape (batch, key_len, embed_dim)
        :param attention_mask: Optional tensor to mask attention weights.
        :return: Output tensor after attention.
        """
        batch_size, query_len, embed_dim = query.size()
        key_len = key.size(1)
        
        # Project queries, keys, and values
        queries = self.query_proj(query)  # (batch, query_len, embed_dim)
        keys = self.key_proj(key)          # (batch, key_len, embed_dim)
        values = self.value_proj(value)    # (batch, key_len, embed_dim)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, query_len, head_dim)
        keys = keys.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)          # (batch, num_heads, key_len, head_dim)
        values = values.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)      # (batch, num_heads, key_len, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale  # (batch, num_heads, query_len, key_len)
        
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, num_heads, query_len, key_len)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, values)  # (batch, num_heads, query_len, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, embed_dim)  # (batch, query_len, embed_dim)
        
        # Final linear projection
        attn_output = self.out_proj(attn_output)  # (batch, query_len, embed_dim)
        
        # Integrate Memory Bank
        # Retrieve memory based on query embeddings
        memory_values = self.memory_bank.retrieve_memory(query.mean(dim=1))  # (batch, top_k, embed_dim)
        
        if memory_values is not None:
            # Project memory values
            memory_queries = self.query_proj(memory_values)  # (batch, top_k, embed_dim)
            memory_keys = self.key_proj(memory_values)      # (batch, top_k, embed_dim)
            memory_values = self.value_proj(memory_values)  # (batch, top_k, embed_dim)
            
            # Reshape for multi-head
            memory_queries = memory_queries.view(batch_size, memory_values.size(1), self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, top_k, head_dim)
            memory_keys = memory_keys.view(batch_size, memory_values.size(1), self.num_heads, self.head_dim).transpose(1, 2)          # (batch, num_heads, top_k, head_dim)
            memory_values = memory_values.view(batch_size, memory_values.size(1), self.num_heads, self.head_dim).transpose(1, 2)      # (batch, num_heads, top_k, head_dim)
            
            # Compute attention scores with memory
            memory_attn_scores = torch.matmul(queries, memory_keys.transpose(-2, -1)) * self.scale  # (batch, num_heads, query_len, top_k)
            
            # Combine with existing attention scores
            combined_attn_scores = attn_scores + memory_attn_scores  # (batch, num_heads, query_len, key_len + top_k)
            
            # Update attention weights
            combined_attn_weights = F.softmax(combined_attn_scores, dim=-1)  # (batch, num_heads, query_len, key_len + top_k)
            combined_attn_weights = F.dropout(combined_attn_weights, p=0.1, training=self.training)
            
            # Concatenate values with memory values
            combined_values = torch.cat([values, memory_values], dim=2)  # (batch, num_heads, key_len + top_k, head_dim)
            
            # Compute combined attention output
            combined_attn_output = torch.matmul(combined_attn_weights, combined_values)  # (batch, num_heads, query_len, head_dim)
            
            # Concatenate heads
            combined_attn_output = combined_attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, embed_dim)  # (batch, query_len, embed_dim)
            
            # Final linear projection
            combined_attn_output = self.out_proj(combined_attn_output)  # (batch, query_len, embed_dim)
            
            attn_output = combined_attn_output  # Override previous attention output
        
        return attn_output  # (batch, query_len, embed_dim)
