import torch
import torch.nn as nn
from models.embedder import TokenEmbedder
from models.exploration import ExploratoryMechanism
from models.rag_scorer import RAGScorer
from models.query_reconstructor import QueryReconstructor

class RAGQueryOptimizer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, top_n=5):
        super(RAGQueryOptimizer, self).__init__()
        self.embedder = TokenEmbedder(vocab_size, embedding_dim)
        self.exploration = ExploratoryMechanism(embedding_dim, top_n)
        self.rag_scorer = RAGScorer(embedding_dim)
        self.query_reconstructor = QueryReconstructor()

    def forward(self, query_tokens, context_tokens):
        query_embeddings = self.embedder(query_tokens)
        context_embeddings = self.embedder(context_tokens)
        
        top_n_distances, top_n_indices = self.exploration(query_embeddings, context_embeddings)
        
        top_n_embeddings = context_embeddings[top_n_indices]
        
        rag_scores = self.rag_scorer(query_embeddings, top_n_embeddings, top_n_distances)
        
        reconstructed_query = self.query_reconstructor(query_tokens, rag_scores)
        
        return reconstructed_query
