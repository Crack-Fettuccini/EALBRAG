import torch
import torch.nn as nn

class ExploratoryMechanism(nn.Module):
    def __init__(self, embedding_dim, top_n=5):
        super(ExploratoryMechanism, self).__init__()
        self.top_n = top_n
        self.latent_query_projection = nn.Linear(embedding_dim, embedding_dim)

    def calculate_distances(self, query_embeddings, context_embeddings):
        query_projected = self.latent_query_projection(query_embeddings)
        distances = torch.cdist(query_projected, context_embeddings, p=2)
        return distances
    
    def forward(self, query_embeddings, context_embeddings):
        distances = self.calculate_distances(query_embeddings, context_embeddings)
        top_n_distances, top_n_indices = torch.topk(distances, self.top_n, dim=-1, largest=False)
        return top_n_distances, top_n_indices
