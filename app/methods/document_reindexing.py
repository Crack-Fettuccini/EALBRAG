import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class DocumentReindexing(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super(DocumentReindexing, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

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

        # Reshape attention weights to (batch_size, num_docs, doc_len)
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
        relevance_scores, sorted_indices = attention_weights.mean(dim=-1).sort(dim=1, descending=True)

        # Gather documents according to sorted indices
        batch_size, num_docs, doc_len, embed_dim = documents.shape
        sorted_indices_exp = sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, doc_len, embed_dim)
        reordered_docs = torch.gather(documents, 1, sorted_indices_exp)
        
        return reordered_docs

    def chunk_documents(self, documents: torch.Tensor, chunk_sizes: List[int]) -> List[torch.Tensor]:
        """
        Split documents into variable-sized chunks.
        
        Args:
            documents: Documents tensor (batch_size, num_docs, doc_len, embed_dim)
            chunk_sizes: List of chunk sizes for each document in the batch
        
        Returns:
            chunked_docs: List of tensors with variable chunk sizes
        """
        batch_size, num_docs, doc_len, embed_dim = documents.size()
        chunked_docs = []

        for batch_idx in range(batch_size):
            batch_chunks = []
            for doc_idx in range(num_docs):
                doc = documents[batch_idx, doc_idx]
                start = 0
                for chunk_size in chunk_sizes[doc_idx]:
                    end = min(start + chunk_size, doc_len)
                    batch_chunks.append(doc[start:end])  # Extract chunk
                    start = end
            chunked_docs.append(torch.cat(batch_chunks, dim=0).unsqueeze(0))  # Combine chunks for batch

        return torch.cat(chunked_docs, dim=0)

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
