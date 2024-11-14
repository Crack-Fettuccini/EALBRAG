import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class DocumentReindexing(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, chunk_size: int = 512):
        super(DocumentReindexing, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size

        # Linear layers for query, key, and value transformations
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        # Multi-head attention module
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def reorder_documents(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        """
        Dynamically reorder documents based on their importance for the query.
        
        Args:
            query: Query tensor (batch_size, query_len, embed_dim)
            documents: Documents tensor (batch_size, num_docs, doc_len, embed_dim)
        
        Returns:
            reordered_documents: Tensor of reordered documents (batch_size, num_docs, doc_len, embed_dim)
        """
        attention_weights = self.compute_attention(query, documents)
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize across the document axis
        reordered_docs = self.apply_attention_reordering(documents, attention_weights)
        return reordered_docs

    def compute_attention(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head attention between the query and documents.
        
        Args:
            query: Query tensor (batch_size, query_len, embed_dim)
            documents: Documents tensor (batch_size, num_docs, doc_len, embed_dim)
        
        Returns:
            attention_weights: Attention weights (batch_size, num_docs, query_len)
        """
        batch_size, num_docs, doc_len, _ = documents.size()
        query_transformed = self.query_linear(query)  # (batch_size, query_len, embed_dim)
        doc_transformed = self.key_linear(documents)  # (batch_size, num_docs, doc_len, embed_dim)

        # Flatten documents to (batch_size, num_docs * doc_len, embed_dim)
        doc_flat = doc_transformed.view(batch_size, num_docs * doc_len, -1)

        # Compute attention weights
        _, attn_weights = self.attn(query_transformed, doc_flat, doc_flat)

        # Reshape attention weights to (batch_size, num_docs, query_len)
        attn_weights = attn_weights.view(batch_size, num_docs, doc_len, query.size(1)).sum(dim=2)
        return attn_weights

    def apply_attention_reordering(self, documents: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Reorder documents based on attention weights.
        
        Args:
            documents: Documents tensor (batch_size, num_docs, doc_len, embed_dim)
            attention_weights: Attention scores (batch_size, num_docs, query_len)
        
        Returns:
            reordered_documents: Reordered documents tensor (batch_size, num_docs, doc_len, embed_dim)
        """
        # Max attention score across query dimension for sorting
        relevance_scores, sorted_indices = attention_weights.mean(dim=-1).sort(dim=1, descending=True)

        # Gather documents according to sorted indices
        batch_size, num_docs, doc_len, embed_dim = documents.shape
        sorted_indices_exp = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, doc_len, embed_dim)
        reordered_docs = torch.gather(documents, 1, sorted_indices_exp)
        
        return reordered_docs

    def chunk_documents(self, documents: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: Documents tensor (batch_size, num_docs, doc_len, embed_dim)
            chunk_size: Chunk size for dividing documents
        
        Returns:
            chunked_docs: Tensor of chunked documents
        """
        chunk_size = chunk_size or self.chunk_size
        batch_size, num_docs, doc_len, embed_dim = documents.shape
        num_chunks = (doc_len + chunk_size - 1) // chunk_size

        # Generate chunked documents
        chunked_docs = torch.cat([documents[:, :, i*chunk_size:(i+1)*chunk_size, :]
                                  for i in range(num_chunks)], dim=2)
        return chunked_docs

    def handle_multimodal_inputs(
        self,
        text_query: torch.Tensor,
        text_documents: torch.Tensor,
        image_documents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Handle both text and image documents with a unified attention mechanism.
        
        Args:
            text_query: Query tensor for text (batch_size, query_len, embed_dim)
            text_documents: Text documents tensor (batch_size, num_docs, doc_len, embed_dim)
            image_documents: Image documents tensor (batch_size, num_docs, img_len, embed_dim)
        
        Returns:
            reordered_text_docs: Reordered text documents tensor
            reordered_image_docs: Reordered image documents tensor
        """
        text_attention_weights = self.compute_attention(text_query, text_documents)
        image_attention_weights = self.compute_attention(text_query, image_documents)

        combined_attention_weights = (text_attention_weights + image_attention_weights) / 2

        reordered_text_docs = self.apply_attention_reordering(text_documents, combined_attention_weights)
        reordered_image_docs = self.apply_attention_reordering(image_documents, combined_attention_weights)

        return reordered_text_docs, reordered_image_docs
