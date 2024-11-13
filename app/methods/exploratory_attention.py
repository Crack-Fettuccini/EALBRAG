import torch
import torch.nn.functional as F

class ExploratoryAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super(ExploratoryAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # Linear layers for transforming input embeddings
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, indexed_docs, hypothetical_docs, lambda_factor=1.0):
        """
        Implements exploratory attention mechanism
        Args:
            query: The query tensor (batch_size, query_len, embed_dim)
            indexed_docs: The indexed documents (batch_size, num_indexed_docs, doc_len, embed_dim)
            hypothetical_docs: The hypothetical documents (batch_size, num_hypothetical_docs, doc_len, embed_dim)
            lambda_factor: The scaling factor for hypothetical documents
        Returns:
            Attention output tensor
        """
        # Transform query, indexed_docs, and hypothetical_docs to the appropriate shape
        query_transformed = self.query_linear(query)  # (batch_size, query_len, embed_dim)
        indexed_docs_transformed = self.key_linear(indexed_docs)  # (batch_size, num_indexed_docs, doc_len, embed_dim)
        hypothetical_docs_transformed = self.key_linear(hypothetical_docs)  # (batch_size, num_hypothetical_docs, doc_len, embed_dim)

        # Compute attention scores for indexed documents
        indexed_attention_scores = self.compute_attention(query_transformed, indexed_docs_transformed)
        
        # Compute attention scores for hypothetical documents
        hypothetical_attention_scores = self.compute_attention(query_transformed, hypothetical_docs_transformed)
        
        # Combine the attention scores
        augmented_attention_scores = lambda_factor * hypothetical_attention_scores + indexed_attention_scores
        
        # Use the augmented attention scores to get the contextually relevant values
        augmented_docs = torch.cat([indexed_docs, hypothetical_docs], dim=1)  # Concatenate indexed and hypothetical docs
        
        # Compute attention output (contextualized values)
        attention_output = self.compute_attention_output(augmented_attention_scores, augmented_docs)
        
        return attention_output

    def compute_attention(self, query, doc):
        """
        Compute attention scores between query and document
        Args:
            query: The query tensor (batch_size, query_len, embed_dim)
            doc: The document tensor (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            Attention scores tensor
        """
        # Compute the dot product between query and document key
        scores = torch.matmul(query, doc.transpose(-2, -1))  # (batch_size, query_len, num_docs, doc_len)
        
        # Normalize scores using softmax to get attention weights
        attention_scores = F.softmax(scores, dim=-1)
        
        return attention_scores

    def compute_attention_output(self, attention_scores, doc):
        """
        Apply attention scores to document values
        Args:
            attention_scores: Attention scores (batch_size, query_len, num_docs, doc_len)
            doc: Document tensor (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            Attention output tensor
        """
        # Compute weighted sum of document values
        attention_output = torch.matmul(attention_scores, doc)  # (batch_size, query_len, num_docs, embed_dim)
        
        # Reduce to a final representation for each query by summing along the query dimension
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

# Initialize the ExploratoryAttention module
exploratory_attention = ExploratoryAttention(embed_dim)

# Forward pass through the exploratory attention mechanism
attention_output = exploratory_attention(query, indexed_docs, hypothetical_docs, lambda_factor=1.0)

print("Attention Output Shape:", attention_output.shape)  # Expected shape: (batch_size, num_docs, embed_dim)
