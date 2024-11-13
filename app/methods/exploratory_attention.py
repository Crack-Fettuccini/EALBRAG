import torch
import torch.nn.functional as F

class DynamicReindexingRAG(torch.nn.Module):
    def __init__(self, embed_dim):
        super(DynamicReindexingRAG, self).__init__()
        self.embed_dim = embed_dim
        
        # Linear layers for transforming query and document embeddings
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, documents, max_steps=10):
        """
        Implements dynamic reindexing and reordering for RAG
        Args:
            query: The query tensor (batch_size, query_len, embed_dim)
            documents: The documents tensor (batch_size, num_docs, doc_len, embed_dim)
            max_steps: Maximum number of token generation steps
        Returns:
            Generated token predictions (batch_size, max_steps, embed_dim)
        """
        generated_tokens = []
        current_query = query
        
        for _ in range(max_steps):
            # Recalculate document importance at each step
            reordered_docs = self.reorder_documents(current_query, documents)
            
            # Create dense matrix from reordered documents
            dense_matrix = self.create_dense_matrix(reordered_docs)
            
            # Compute the next token's attention output
            attention_output = self.compute_attention(current_query, dense_matrix)
            
            # Simulate token generation and update the query
            generated_tokens.append(attention_output)
            
            # Update query based on current context (simple mean pooling for this example)
            current_query = self.update_query(current_query, attention_output)
        
        return torch.stack(generated_tokens, dim=1)

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
        
        # Calculate probability for each document
        probabilities = torch.exp(attention_scores)
        
        # Sort documents by their relevance (descending order)
        sorted_indices = torch.argsort(probabilities, dim=1, descending=True)
        
        # Reorder the documents based on the sorted indices
        reordered_docs = torch.gather(documents, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, documents.size(2), documents.size(3)))
        
        return reordered_docs

    def create_dense_matrix(self, documents):
        """
        Creates a dense matrix from reordered documents
        Args:
            documents: Reordered documents tensor (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            Dense matrix tensor (batch_size, num_docs, embed_dim)
        """
        # Flatten the document matrix to make it suitable for attention mechanism
        dense_matrix = documents.view(documents.size(0), -1, self.embed_dim)
        return dense_matrix

    def compute_attention(self, query, dense_matrix):
        """
        Compute attention output for the next token generation
        Args:
            query: The query tensor (batch_size, query_len, embed_dim)
            dense_matrix: The dense matrix of documents (batch_size, num_docs, embed_dim)
        Returns:
            Attention output tensor (batch_size, embed_dim)
        """
        # Compute attention scores between query and dense matrix using dot-product attention
        attention_scores = self.compute_attention_scores(query, dense_matrix)
        
        # Normalize attention scores using softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute weighted sum of documents based on attention weights
        attention_output = torch.matmul(attention_weights, dense_matrix)
        
        return attention_output

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

    def update_query(self, query, attention_output):
        """
        Update query based on generated token context (simple mean pooling)
        Args:
            query: The current query tensor (batch_size, query_len, embed_dim)
            attention_output: The attention output tensor (batch_size, embed_dim)
        Returns:
            Updated query tensor (batch_size, query_len, embed_dim)
        """
        # For simplicity, update query by concatenating and pooling the attention output
        updated_query = (query.mean(dim=1) + attention_output) / 2
        return updated_query.unsqueeze(1).expand_as(query)


# Example Usage:

batch_size = 2
query_len = 3
doc_len = 4
embed_dim = 5
num_docs = 6
max_steps = 10

# Example input data
query = torch.randn(batch_size, query_len, embed_dim)  # (batch_size, query_len, embed_dim)
documents = torch.randn(batch_size, num_docs, doc_len, embed_dim)  # (batch_size, num_docs, doc_len, embed_dim)

# Initialize the DynamicReindexingRAG module
rag_model = DynamicReindexingRAG(embed_dim)

# Generate response iteratively with document reordering
generated_tokens = rag_model(query, documents, max_steps=max_steps)

print("Generated Tokens Shape:", generated_tokens.shape)  # Expected shape: (batch_size, max_steps, embed_dim)
