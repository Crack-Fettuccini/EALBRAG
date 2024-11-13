import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentReindexing(nn.Module):
    def __init__(self, embed_dim, num_heads=8, chunk_size=512):
        super(DocumentReindexing, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        
        # Linear layers for transforming query and document embeddings
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def reorder_documents(self, query, documents):
        """
        Dynamically reorder documents based on their importance for the next token generation.
        Args:
            query: The current query tensor (batch_size, query_len, embed_dim)
            documents: The documents tensor (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            reordered_documents: Reordered documents tensor (batch_size, num_docs, doc_len, embed_dim)
        """
        # Step 1: Compute multi-head attention scores
        attention_weights = self.compute_attention(query, documents)
        
        # Step 2: Apply softmax and reorder documents based on computed attention weights
        attention_weights = F.softmax(attention_weights, dim=-1)  # Normalize across document axis
        reordered_docs = self.apply_attention_reordering(documents, attention_weights)
        
        return reordered_docs

    def compute_attention(self, query, documents):
        """
        Compute multi-head attention between query and documents.
        Args:
            query: The query tensor (batch_size, query_len, embed_dim)
            documents: The documents tensor (batch_size, num_docs, doc_len, embed_dim)
        Returns:
            attention_weights: The attention weights between query and documents (batch_size, query_len, num_docs)
        """
        # Transform query and documents using linear layers
        query_transformed = self.query_linear(query)  # (batch_size, query_len, embed_dim)
        doc_transformed = self.key_linear(documents)  # (batch_size, num_docs, doc_len, embed_dim)

        # Reshaping for multi-head attention
        query_reshaped = query_transformed.permute(1, 0, 2)  # (query_len, batch_size, embed_dim)
        doc_reshaped = doc_transformed.view(documents.size(0), documents.size(1)*documents.size(2), -1)  # (batch_size, num_docs * doc_len, embed_dim)

        # Apply multi-head attention
        attn_output, attn_weights = self.attn(query_reshaped, doc_reshaped, doc_reshaped)
        
        # Attention weights are returned as (batch_size, num_docs, query_len)
        attn_weights = attn_weights.permute(1, 2, 0)  # Transpose to (batch_size, num_docs, query_len)
        
        return attn_weights

    def apply_attention_reordering(self, documents, attention_weights):
        """
        Reorder documents based on attention weights (sorted by relevance to the query).
        Args:
            documents: The documents tensor (batch_size, num_docs, doc_len, embed_dim)
            attention_weights: The attention scores (batch_size, query_len, num_docs)
        Returns:
            reordered_documents: The reordered documents based on attention scores (batch_size, num_docs, doc_len, embed_dim)
        """
        # Select the most relevant documents based on attention weights
        # We calculate the most relevant documents by considering the max attention score across the query
        attention_scores, sorted_indices = attention_weights.max(dim=2)  # (batch_size, num_docs)
        
        # Reorder the documents based on the sorted attention weights
        reordered_docs = torch.gather(documents, 1, sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, documents.size(2), documents.size(3)))
        
        return reordered_docs

    def chunk_documents(self, documents, chunk_size=None):
        """
        Split documents into smaller chunks of a specified size.
        Args:
            documents: The documents tensor (batch_size, num_docs, doc_len, embed_dim)
            chunk_size: The chunk size for dividing documents (default to self.chunk_size)
        Returns:
            chunked_docs: A tensor of chunked documents (batch_size, num_chunks, chunk_size, embed_dim)
        """
        chunk_size = chunk_size or self.chunk_size
        num_chunks = (documents.size(2) + chunk_size - 1) // chunk_size  # Calculate number of chunks
        chunked_docs = []
        
        for i in range(num_chunks):
            chunk = documents[:, :, i * chunk_size: (i + 1) * chunk_size, :]
            chunked_docs.append(chunk)
        
        return torch.cat(chunked_docs, dim=2)  # (batch_size, num_docs, chunked_len, embed_dim)

    def handle_multimodal_inputs(self, text_query, text_documents, image_documents):
        """
        Handle both textual and visual documents by applying attention across multiple modalities.
        Args:
            text_query: The query tensor for text (batch_size, query_len, embed_dim)
            text_documents: The text documents tensor (batch_size, num_docs, doc_len, embed_dim)
            image_documents: The image documents tensor (batch_size, num_docs, img_len, embed_dim)
        Returns:
            reordered_text_docs: Reordered text documents based on attention weights
            reordered_image_docs: Reordered image documents based on attention weights
        """
        text_attention_weights = self.compute_attention(text_query, text_documents)
        image_attention_weights = self.compute_attention(text_query, image_documents)
        
        # Combine text and image attention weights (e.g., averaging or concatenation)
        combined_attention_weights = (text_attention_weights + image_attention_weights) / 2
        
        # Reorder text and image documents based on combined attention
        reordered_text_docs = self.apply_attention_reordering(text_documents, combined_attention_weights)
        reordered_image_docs = self.apply_attention_reordering(image_documents, combined_attention_weights)
        
        return reordered_text_docs, reordered_image_docs


# Example Usage:
# Define embedding dimensions and other parameters
embed_dim = 256
batch_size = 4
num_docs = 10
doc_len = 100
query_len = 5

# Initialize the Document Reindexing model
model = DocumentReindexing(embed_dim)

# Generate random data for query and documents
query = torch.randn(batch_size, query_len, embed_dim)  # (batch_size, query_len, embed_dim)
documents = torch.randn(batch_size, num_docs, doc_len, embed_dim)  # (batch_size, num_docs, doc_len, embed_dim)

# Reorder documents based on attention mechanism
reordered_docs = model.reorder_documents(query, documents)
print(reordered_docs.shape)  # (batch_size, num_docs, doc_len, embed_dim)
