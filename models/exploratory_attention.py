import torch
import torch.nn.functional as F

class AttentionMechanism(torch.nn.Module):
    def __init__(self, embed_dim):
        super(AttentionMechanism, self).__init__()
        self.embed_dim = embed_dim
        
        # Linear layers for transforming query, document (key), and document (value) embeddings
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, documents):
        """
        Forward pass for the attention mechanism.
        Args:
            query (Tensor): The query tensor of shape (batch_size, query_len, embed_dim)
            documents (Tensor): The document tensor of shape (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            Tensor: The attention output tensor of shape (batch_size, embed_dim)
        """
        # Step 1: Compute the attention output based on the query and documents
        attention_output = self.compute_attention(query, documents)
        return attention_output

    def compute_attention(self, query, documents):
        """
        Compute the attention mechanism using the query and the documents.
        Args:
            query (Tensor): The query tensor of shape (batch_size, query_len, embed_dim)
            documents (Tensor): The documents tensor of shape (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            Tensor: The weighted sum of document embeddings, which is the attention output
        """
        # Step 2: Compute the attention scores between the query and documents using dot-product attention
        attention_scores = self.compute_attention_scores(query, documents)
        
        # Step 3: Apply softmax to the attention scores to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize over documents
        
        # Step 4: Compute the weighted sum of document vectors based on attention weights
        # documents: (batch_size, num_docs, doc_len, embed_dim)
        # attention_weights: (batch_size, query_len, num_docs)
        weighted_documents = torch.matmul(attention_weights, documents)  # (batch_size, query_len, embed_dim)
        
        # Step 5: Aggregate the attention outputs (sum across the query_len dimension)
        attention_output = weighted_documents.sum(dim=1)  # (batch_size, embed_dim)
        
        return attention_output

    def compute_attention_scores(self, query, documents):
        """
        Compute attention scores between the query and documents.
        Args:
            query (Tensor): The query tensor of shape (batch_size, query_len, embed_dim)
            documents (Tensor): The documents tensor of shape (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            Tensor: The attention scores tensor of shape (batch_size, query_len, num_docs)
        """
        # Step 6: Transform query and documents using linear layers to match the attention mechanism
        query_transformed = self.query_linear(query)  # (batch_size, query_len, embed_dim)
        doc_transformed = self.key_linear(documents)  # (batch_size, num_docs, doc_len, embed_dim)
        
        # Step 7: Compute the attention scores using dot product between transformed query and document (key)
        # query_transformed: (batch_size, query_len, embed_dim)
        # doc_transformed: (batch_size, num_docs, embed_dim)
        # Attention score calculation (query_len, num_docs)
        scores = torch.matmul(query_transformed, doc_transformed.transpose(-2, -1))  # (batch_size, query_len, num_docs)
        
        return scores

