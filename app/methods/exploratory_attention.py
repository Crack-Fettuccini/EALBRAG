import torch
import torch.nn.functional as F

class AttentionMechanism(torch.nn.Module):
    def __init__(self, embed_dim):
        super(AttentionMechanism, self).__init__()
        self.embed_dim = embed_dim
        
        # Linear layers for transforming query and document embeddings
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, documents):
        """
        Computes the attention output for the next token generation
        Args:
            query: The query tensor (batch_size, query_len, embed_dim)
            documents: The reordered documents tensor (batch_size, num_docs, embed_dim)
        Returns:
            Attention output tensor (batch_size, embed_dim)
        """
        # Compute the attention output based on the query and documents
        attention_output = self.compute_attention(query, documents)
        
        return attention_output

    def compute_attention(self, query, documents):
        """
        Compute attention output for the next token generation
        Args:
            query: The query tensor (batch_size, query_len, embed_dim)
            documents: The documents tensor (batch_size, num_docs, embed_dim)
        Returns:
            Attention output tensor (batch_size, embed_dim)
        """
        # Compute the attention scores between query and dense matrix using dot-product attention
        attention_scores = self.compute_attention_scores(query, documents)
        
        # Normalize attention scores using softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute weighted sum of documents based on attention weights
        attention_output = torch.matmul(attention_weights, documents)
        
        return attention_output

    def compute_attention_scores(self, query, documents):
        """
        Compute attention scores between query and documents
        Args:
            query: The query tensor (batch_size, query_len, embed_dim)
            documents: The documents tensor (batch_size, num_docs, embed_dim)
        Returns:
            Attention scores tensor (batch_size, query_len, num_docs)
        """
        # Transform query and documents to the same dimension for dot-product
        query_transformed = self.query_linear(query)  # (batch_size, query_len, embed_dim)
        doc_transformed = self.key_linear(documents)  # (batch_size, num_docs, embed_dim)
        
        # Compute the dot-product between query and document vectors
        scores = torch.matmul(query_transformed, doc_transformed.transpose(-2, -1))  # (batch_size, query_len, num_docs)
        
        # Normalize scores using softmax to get attention weights
        attention_scores = scores
        
        return attention_scores
