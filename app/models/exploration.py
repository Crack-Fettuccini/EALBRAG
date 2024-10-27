import torch
import torch.nn as nn
import torch.nn.functional as F

class ExploratoryMechanism(nn.Module):
    def __init__(self, embedding_dim, top_n=5, memory_bank=None, distance_metric='euclidean', normalize=False, use_gating=True):
        """
        :param embedding_dim: Dimensionality of token embeddings
        :param top_n: Number of closest tokens to consider for exploration
        :param memory_bank: Instance of MemoryBank to integrate memory
        :param distance_metric: Metric to use for distance ('euclidean', 'cosine')
        :param normalize: Whether to normalize embeddings before computing distances
        :param use_gating: Whether to use gating for adaptive memory-context integration
        """
        super(ExploratoryMechanism, self).__init__()
        self.top_n = top_n
        self.normalize = normalize
        self.distance_metric = distance_metric
        self.latent_query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.memory_bank = memory_bank  # Optional memory integration
        self.use_gating = use_gating
        
        # Gating mechanism for adaptive memory-context integration
        if self.use_gating:
            self.gating_layer = nn.Linear(embedding_dim, 1)
        
        # Learned distance metric (bilinear form)
        self.distance_projection = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        nn.init.xavier_uniform_(self.distance_projection)

        # Optional dropout
        self.dropout = nn.Dropout(0.1)

    def calculate_distances(self, query_embeddings: torch.Tensor, context_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate the pairwise distances between query tokens and context tokens.
        :param query_embeddings: Embedded query tokens (batch, seq_len, embed_dim)
        :param context_embeddings: Embedded context tokens (batch, context_seq_len, embed_dim)
        :return: Distances between query and context embeddings (batch, seq_len, context_seq_len)
        """
        # Optionally normalize the embeddings
        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
            context_embeddings = F.normalize(context_embeddings, p=2, dim=-1)

        # Project query embeddings
        query_projected = self.latent_query_projection(query_embeddings)  # (batch, seq_len, embed_dim)

        # Calculate distances based on the chosen metric
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(query_projected, context_embeddings, p=2)  # (batch, seq_len, context_seq_len)
        elif self.distance_metric == 'cosine':
            distances = 1 - torch.bmm(query_projected, context_embeddings.transpose(1, 2))  # (batch, seq_len, context_seq_len)
        else:
            # Learned bilinear distance metric
            query_projected = query_projected @ self.distance_projection  # (batch, seq_len, embed_dim)
            distances = torch.bmm(query_projected, context_embeddings.transpose(1, 2))  # (batch, seq_len, context_seq_len)

        return distances

    def forward(self, query_embeddings: torch.Tensor, context_embeddings: torch.Tensor, memory_embeddings: torch.Tensor = None):
        """
        Forward method to explore the high-dimensional space, considering memory.
        :param query_embeddings: Embeddings of the query tokens (batch, seq_len, embed_dim)
        :param context_embeddings: Embeddings of the context tokens (batch, context_seq_len, embed_dim)
        :param memory_embeddings: Embeddings retrieved from memory (batch, top_k, embed_dim)
        :return: Top-N closest distances and their corresponding indices (batch, seq_len, top_n)
        """
        distances = self.calculate_distances(query_embeddings, context_embeddings)  # (batch, seq_len, context_seq_len)

        # Incorporate memory embeddings if provided
        if memory_embeddings is not None:
            memory_embeddings = memory_embeddings.view(memory_embeddings.size(0), -1, memory_embeddings.size(-1))  # (batch, top_k, embed_dim)
            distances_memory = self.calculate_distances(query_embeddings, memory_embeddings)  # (batch, seq_len, top_k)

            if self.use_gating:
                # Use gating to decide how much to weigh context vs memory
                context_gate = torch.sigmoid(self.gating_layer(query_embeddings))  # (batch, seq_len, 1)
                distances = distances * context_gate + distances_memory * (1 - context_gate)
            else:
                distances = torch.cat([distances, distances_memory], dim=-1)  # (batch, seq_len, context_seq_len + top_k)

        # Efficient top-N calculation
        top_n_distances, top_n_indices = torch.topk(distances, self.top_n, dim=-1, largest=False)  # (batch, seq_len, top_n)

        return top_n_distances, top_n_indices
