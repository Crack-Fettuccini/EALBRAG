import torch

class DocumentReindexing(torch.nn.Module):
    def __init__(self, embed_dim):
        super(DocumentReindexing, self).__init__()
        self.embed_dim = embed_dim
        
        # Linear layers for transforming query and document embeddings
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        
    def reorder_documents(self, query, documents):
        """
        Dynamically reorder documents based on their importance for the next token generation
        Args:
            query: The current query tensor (batch_size, query_len, embed_dim)
            documents: The documents tensor (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            Reordered documents tensor (batch_size, num_docs, doc_len, embed_dim)
        """
        attention_scores = self.compute_attention_scores(query, documents)
        
        # Calculate probability for each document based on its attention score
        probabilities = torch.exp(attention_scores)
        
        # Sort documents by their relevance (descending order)
        sorted_indices = torch.argsort(probabilities, dim=1, descending=True)
        
        # Reorder the documents based on the sorted indices
        reordered_docs = torch.gather(documents, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, documents.size(2), documents.size(3)))
        
        return reordered_docs

    def compute_attention_scores(self, query, documents):
        """
        Compute attention scores between query and documents
        Args:
            query: The query tensor (batch_size, query_len, embed_dim)
            documents: The documents tensor (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            Attention scores tensor (batch_size, query_len, num_docs)
        """
        # Transform query and documents to the same dimension for dot-product
        query_transformed = self.query_linear(query)  # (batch_size, query_len, embed_dim)
        doc_transformed = self.key_linear(documents)  # (batch_size, num_docs, doc_len, embed_dim)
        
        # Compute the dot-product between query and document vectors
        scores = torch.matmul(query_transformed, doc_transformed.transpose(-2, -1))  # (batch_size, query_len, num_docs, doc_len)
        
        # Normalize scores using softmax to get attention weights
        attention_scores = scores.sum(dim=-1)  # Summing over doc_len to get a scalar score
        
        return attention_scores
