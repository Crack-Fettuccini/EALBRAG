# models/retriever.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Retriever:
    def __init__(self, embedder, knowledge_base_embeddings, device, top_k=5):
        """
        Initialize the retriever with an embedder and knowledge base.
        :param embedder: A module to embed documents and queries
        :param knowledge_base_embeddings: Precomputed embeddings for the knowledge base documents (num_docs, embed_dim)
        :param device: torch.device
        :param top_k: Number of top documents to retrieve
        """
        self.embedder = embedder
        self.knowledge_base_embeddings = knowledge_base_embeddings.to(device)  # (num_docs, embed_dim)
        self.device = device
        self.top_k = top_k
    
    def embed_documents(self, documents):
        """
        Embed a list of documents.
        :param documents: List of document strings
        :return: Tensor of shape (num_docs, embed_dim)
        """
        with torch.no_grad():
            inputs = torch.stack([self.embedder(torch.tensor(doc)) for doc in documents]).to(self.device)
            embeddings = inputs.mean(dim=1)  # (num_docs, embed_dim)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def retrieve(self, query_embeddings):
        """
        Retrieve top-k documents based on query embeddings.
        :param query_embeddings: Tensor of shape (batch, embed_dim)
        :return: List of top-k document indices for each query in the batch
        """
        # Normalize query embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)  # (batch, embed_dim)
        
        # Compute cosine similarity with knowledge base
        similarities = torch.matmul(query_embeddings, self.knowledge_base_embeddings.T)  # (batch, num_docs)
        
        # Get top-k indices
        topk_values, topk_indices = torch.topk(similarities, self.top_k, dim=-1)  # (batch, top_k)
        
        return topk_indices  # (batch, top_k)
