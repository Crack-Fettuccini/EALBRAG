# models/exploration.py
import torch
import torch.nn as nn

class ExploratoryMechanism(nn.Module):
    def __init__(self, embedding_dim, top_n=5, memory_bank=None):
        """
        :param embedding_dim: Dimensionality of token embeddings
        :param top_n: Number of closest tokens to consider for exploration
        :param memory_bank: Instance of MemoryBank to integrate memory
        """
        super(ExploratoryMechanism, self).__init__()
        self.top_n = top_n
        self.latent_query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.memory_bank = memory_bank  # Optional memory integration

    def calculate_distances(self, query_embeddings, context_embeddings):
        """
        Calculate the pairwise distances between query tokens and context tokens.
        :param query_embeddings: Embedded query tokens (batch, seq_len, embed_dim)
        :param context_embeddings: Embedded context tokens (batch, context_seq_len, embed_dim)
        :return: Distances between query and context embeddings (batch, seq_len, context_seq_len)
        """
        # Project query embeddings
        query_projected = self.latent_query_projection(query_embeddings)  # (batch, seq_len, embed_dim)
        # Compute pairwise Euclidean distances
        distances = torch.cdist(query_projected, context_embeddings, p=2)  # (batch, seq_len, context_seq_len)
        return distances

    def forward(self, query_embeddings, context_embeddings, memory_embeddings=None):
        """
        Forward method to explore the high-dimensional space, considering memory.
        :param query_embeddings: Embeddings of the query tokens (batch, seq_len, embed_dim)
        :param context_embeddings: Embeddings of the context tokens (batch, context_seq_len, embed_dim)
        :param memory_embeddings: Embeddings retrieved from memory (batch, top_k, embed_dim)
        :return: Top-N closest distances and their corresponding indices (batch, seq_len, top_n)
        """
        distances = self.calculate_distances(query_embeddings, context_embeddings)  # (batch, seq_len, context_seq_len)

        if memory_embeddings is not None:
            # Concatenate memory embeddings to context
            memory_embeddings = memory_embeddings.view(memory_embeddings.size(0), -1, memory_embeddings.size(-1))  # (batch, top_k, embed_dim)
            distances_memory = torch.cdist(
                self.latent_query_projection(query_embeddings),
                memory_embeddings,
                p=2
            )  # (batch, seq_len, top_k)
            # Concatenate distances
            distances = torch.cat([distances, distances_memory], dim=-1)  # (batch, seq_len, context_seq_len + top_k)

        # For each query token, find the top-n closest context (including memory) tokens
        top_n_distances, top_n_indices = torch.topk(distances, self.top_n, dim=-1, largest=False)  # (batch, seq_len, top_n)
        return top_n_distances, top_n_indices
