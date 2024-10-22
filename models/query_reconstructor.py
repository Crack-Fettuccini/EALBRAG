import torch
import torch.nn as nn

class QueryReconstructor(nn.Module):
    def __init__(self, ignore_special_tokens=True, pad_token_id=0):
        """
        Initialize the QueryReconstructor module.
        
        :param ignore_special_tokens: Whether to ignore special tokens like [PAD], [CLS], [SEP].
        :param pad_token_id: The token ID of the padding token (default is 0 for [PAD] in many models).
        """
        super(QueryReconstructor, self).__init__()
        self.ignore_special_tokens = ignore_special_tokens
        self.pad_token_id = pad_token_id

    def reconstruct_query(self, query_tokens, rag_scores, attention_mask=None):
        """
        Reconstruct the query by reordering tokens based on their RAG scores.
        
        :param query_tokens: Tensor of shape (batch_size, seq_len) representing the token IDs of queries.
        :param rag_scores: Tensor of shape (batch_size, seq_len) representing the RAG scores for each token.
        :param attention_mask: Optional tensor of shape (batch_size, seq_len), where 1 indicates valid tokens, and 0 indicates padding.
        
        :return: Tensor of shape (batch_size, seq_len) with the query tokens reordered based on RAG scores.
        """
        # If there's an attention mask, exclude special tokens (like padding) from being reordered
        if self.ignore_special_tokens and attention_mask is not None:
            rag_scores = rag_scores.masked_fill(attention_mask == 0, float('-inf'))  # Mask out special tokens
        
        # Sort tokens based on the rag_scores in descending order
        sorted_indices = torch.argsort(rag_scores, dim=-1, descending=True)

        # Reorder the tokens based on the sorted indices
        reordered_query = torch.gather(query_tokens, dim=1, index=sorted_indices)

        return reordered_query

    def forward(self, query_tokens, rag_scores, attention_mask=None):
        """
        Forward pass to reconstruct the query.
        
        :param query_tokens: Tensor of shape (batch_size, seq_len) representing token IDs.
        :param rag_scores: Tensor of shape (batch_size, seq_len) representing RAG scores for each token.
        :param attention_mask: Optional tensor of shape (batch_size, seq_len) to mask special tokens.
        
        :return: Reconstructed queries with reordered tokens.
        """
        return self.reconstruct_query(query_tokens, rag_scores, attention_mask)

# Example usage:
if __name__ == "__main__":
    # Example input
    query_tokens = torch.tensor([[101, 3452, 2054, 999, 102, 0], [101, 2129, 2003, 1996, 3185, 102]])
    rag_scores = torch.tensor([[0.2, 0.5, 0.1, 0.8, 0.4, -float('inf')], [0.3, 0.7, 0.4, 0.9, 0.2, 0.6]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])  # Masking the padding tokens in the first query

    model = QueryReconstructor(ignore_special_tokens=True, pad_token_id=0)
    reordered_queries = model(query_tokens, rag_scores, attention_mask)

    print("Reordered Queries:\n", reordered_queries)
