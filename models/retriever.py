import torch
import torch.nn as nn
import torch.nn.functional as F

class Retriever:
    def __init__(self, embedder, knowledge_base_embeddings, tokenizer, device, top_k=5, max_length=512):
        """
        Initialize the retriever with an embedder and knowledge base.
        :param embedder: A module to embed documents and queries (can be transformer or custom model)
        :param knowledge_base_embeddings: Precomputed embeddings for the knowledge base documents (num_docs, embed_dim)
        :param tokenizer: Tokenizer to process text queries and documents
        :param device: torch.device
        :param top_k: Number of top documents to retrieve for each query
        :param max_length: Maximum length for tokenization (important for chunking long queries or documents)
        """
        self.embedder = embedder
        self.knowledge_base_embeddings = knowledge_base_embeddings.to(device)  # (num_docs, embed_dim)
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = top_k
        self.max_length = max_length

    def embed_text(self, texts):
        """
        Tokenize and embed a batch of text documents or queries.
        :param texts: List of text strings to embed
        :return: Tensor of shape (num_texts, embed_dim) containing normalized embeddings
        """
        with torch.no_grad():
            # Tokenize texts and handle truncation if they exceed max_length
            tokens = [self.tokenizer.encode(text, return_tensors='pt', max_length=self.max_length, truncation=True).to(self.device) for text in texts]
            
            # Pad the tokenized sequences to have consistent shape
            tokens_padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True).to(self.device)
            
            # Generate embeddings using the embedder model
            if hasattr(self.embedder, 'forward'):
                embeddings = self.embedder(tokens_padded).last_hidden_state.mean(dim=1)  # Mean pooling over sequence length
            else:
                embeddings = self.embedder(tokens_padded)  # If it's a custom embedder, assume it outputs directly
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def embed_documents(self, documents):
        """
        Embed a list of documents.
        :param documents: List of document strings
        :return: Tensor of shape (num_docs, embed_dim)
        """
        return self.embed_text(documents)

    def embed_queries(self, queries):
        """
        Embed a list of queries.
        :param queries: List of query strings
        :return: Tensor of shape (num_queries, embed_dim)
        """
        return self.embed_text(queries)

    def retrieve(self, query_embeddings):
        """
        Retrieve top-k documents based on query embeddings.
        :param query_embeddings: Tensor of shape (batch, embed_dim)
        :return: List of top-k document indices for each query in the batch
        """
        # Normalize query embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)  # (batch, embed_dim)
        
        # Compute cosine similarity with knowledge base embeddings
        similarities = torch.matmul(query_embeddings, self.knowledge_base_embeddings.T)  # (batch, num_docs)
        
        # Get top-k indices for each query
        topk_values, topk_indices = torch.topk(similarities, self.top_k, dim=-1)  # (batch, top_k)
        
        return topk_indices  # (batch, top_k)

    def retrieve_from_text(self, queries):
        """
        Embed queries and retrieve top-k documents.
        :param queries: List of query strings
        :return: List of top-k document indices for each query
        """
        # Embed the queries first
        query_embeddings = self.embed_queries(queries)
        
        # Retrieve the top-k documents based on query embeddings
        return self.retrieve(query_embeddings)

    def get_top_k_documents(self, queries, document_metadata):
        """
        Given a list of queries, retrieve and return the metadata of the top-k matching documents.
        :param queries: List of query strings
        :param document_metadata: List of document metadata (filenames, paths, etc.) matching the knowledge base embeddings
        :return: List of lists containing top-k document metadata for each query
        """
        # Get the top-k document indices for each query
        top_k_indices = self.retrieve_from_text(queries)
        
        # Map the indices back to document metadata
        top_k_metadata = []
        for indices in top_k_indices:
            retrieved_docs = [document_metadata[idx] for idx in indices]
            top_k_metadata.append(retrieved_docs)
        
        return top_k_metadata
