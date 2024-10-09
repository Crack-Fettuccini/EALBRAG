import torch
import torch.nn as nn

class QueryReconstructor(nn.Module):
    def __init__(self):
        super(QueryReconstructor, self).__init__()

    def reconstruct_query(self, query_tokens, rag_scores):
        sorted_indices = torch.argsort(rag_scores, descending=True)
        reordered_query = query_tokens[sorted_indices]
        return reordered_query

    def forward(self, query_tokens, rag_scores):
        return self.reconstruct_query(query_tokens, rag_scores)
