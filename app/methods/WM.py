import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkingMemory(nn.Module):
    def __init__(self, embed_dim):
        super(WorkingMemory, self).__init__()
        self.embed_dim = embed_dim
        
        # LSTM layer to manage memory state
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        
    def forward(self, X, prev_memory_state):
        """
        Updates the working memory using LSTM.
        Args:
            X: The input tensor at time step t (batch_size, seq_len, embed_dim)
            prev_memory_state: The previous memory state (h_0, c_0) from previous step
        Returns:
            Updated memory state
        """
        # Pass input through LSTM, get updated memory states (h_t, c_t)
        memory_state, (h_t, c_t) = self.lstm(X, prev_memory_state)
        
        return memory_state, (h_t, c_t)


class MemoryAugmentedAttention(nn.Module):
    def __init__(self, embed_dim):
        super(MemoryAugmentedAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # Linear layers for transforming inputs
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, indexed_docs, hypothetical_docs, memory_state, lambda_factor=1.0):
        """
        Implements memory-augmented attention.
        Args:
            query: Query tensor (batch_size, query_len, embed_dim)
            indexed_docs: Indexed document tensor (batch_size, num_indexed_docs, doc_len, embed_dim)
            hypothetical_docs: Hypothetical document tensor (batch_size, num_hypothetical_docs, doc_len, embed_dim)
            memory_state: The memory-enhanced hidden state from LSTM
            lambda_factor: Weight for hypothetical documents in the attention computation
        Returns:
            Attention output tensor
        """
        # Apply linear transformations to query, docs, and memory
        query_transformed = self.query_linear(query)  # (batch_size, query_len, embed_dim)
        indexed_docs_transformed = self.key_linear(indexed_docs)  # (batch_size, num_indexed_docs, doc_len, embed_dim)
        hypothetical_docs_transformed = self.key_linear(hypothetical_docs)  # (batch_size, num_hypothetical_docs, doc_len, embed_dim)

        # Combine query with memory state
        query_memory_enriched = query_transformed + memory_state  # (batch_size, query_len, embed_dim)
        
        # Compute attention scores for indexed documents
        indexed_attention_scores = self.compute_attention(query_memory_enriched, indexed_docs_transformed)
        
        # Compute attention scores for hypothetical documents
        hypothetical_attention_scores = self.compute_attention(query_memory_enriched, hypothetical_docs_transformed)
        
        # Combine both attention scores with the lambda factor
        augmented_attention_scores = lambda_factor * hypothetical_attention_scores + indexed_attention_scores
        
        # Combine indexed and hypothetical documents
        augmented_docs = torch.cat([indexed_docs, hypothetical_docs], dim=1)
        
        # Compute attention output using augmented scores
        attention_output = self.compute_attention_output(augmented_attention_scores, augmented_docs)
        
        return attention_output

    def compute_attention(self, query, doc):
        """
        Compute attention scores using scaled dot-product attention.
        Args:
            query: Query tensor (batch_size, query_len, embed_dim)
            doc: Document tensor (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            Attention scores tensor
        """
        # Compute dot product between query and document keys
        scores = torch.matmul(query, doc.transpose(-2, -1))  # (batch_size, query_len, num_docs, doc_len)
        
        # Normalize scores with softmax to get attention weights
        attention_scores = F.softmax(scores, dim=-1)
        
        return attention_scores

    def compute_attention_output(self, attention_scores, doc):
        """
        Apply attention scores to documents to get weighted sum of values.
        Args:
            attention_scores: Attention scores (batch_size, query_len, num_docs, doc_len)
            doc: Document tensor (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            Attention output tensor
        """
        # Compute weighted sum of document values using attention scores
        attention_output = torch.matmul(attention_scores, doc)  # (batch_size, query_len, num_docs, embed_dim)
        
        # Summing over query_len to get the final representation for each query
        attention_output = attention_output.sum(dim=1)  # (batch_size, num_docs, embed_dim)
        
        return attention_output


# Example Usage:

batch_size = 2
query_len = 3
doc_len = 4
embed_dim = 5
num_indexed_docs = 6
num_hypothetical_docs = 4

# Example input data
query = torch.randn(batch_size, query_len, embed_dim)  # (batch_size, query_len, embed_dim)
indexed_docs = torch.randn(batch_size, num_indexed_docs, doc_len, embed_dim)  # (batch_size, num_indexed_docs, doc_len, embed_dim)
hypothetical_docs = torch.randn(batch_size, num_hypothetical_docs, doc_len, embed_dim)  # (batch_size, num_hypothetical_docs, doc_len, embed_dim)

# Initialize LSTM-based working memory module
working_memory = WorkingMemory(embed_dim)

# Initial memory state (h_0, c_0) is initialized as zeros
initial_memory_state = (torch.zeros(batch_size, query_len, embed_dim), torch.zeros(batch_size, query_len, embed_dim))

# Process input through the working memory
memory_state, updated_memory_state = working_memory(query, initial_memory_state)

# Initialize memory-augmented attention module
memory_augmented_attention = MemoryAugmentedAttention(embed_dim)

# Forward pass through the memory-augmented attention mechanism
attention_output = memory_augmented_attention(query, indexed_docs, hypothetical_docs, memory_state, lambda_factor=1.0)

print("Attention Output Shape:", attention_output.shape)  # Expected shape: (batch_size, num_docs, embed_dim)
