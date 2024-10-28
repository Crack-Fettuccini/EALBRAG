import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class AttentionAnalyzer:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)

    def get_most_attended_tokens(self, input_text, top_k=3):
        """
        Analyzes attention scores for each token's generation step.
        
        Parameters:
            input_text (str): The input text to analyze.
            top_k (int): The number of top-attended tokens to retrieve at each generation step.
        
        Returns:
            dict: A dictionary where each key is a token position, and values are lists of top attended tokens.
        """
        # Tokenize the input and prepare for model
        inputs = self.tokenizer(input_text, return_tensors="pt")
        attention_scores_per_token = {}

        # Run the model and get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Loop through each layer's attention outputs
        for layer_idx, layer_attention in enumerate(outputs.attentions):
            # `layer_attention` has shape (batch_size, num_heads, seq_len, seq_len)
            # We focus on the last hidden layer's attentions, reducing attention scores across heads.
            avg_attention = layer_attention.mean(dim=1)  # Shape: (seq_len, seq_len)
            
            for token_idx in range(avg_attention.shape[1]):
                # Get attention scores for the current token
                token_attention_scores = avg_attention[0, token_idx]
                
                # Identify the top `k` most attended tokens
                top_indices = torch.topk(token_attention_scores, top_k).indices
                top_tokens = [self.tokenizer.decode(inputs["input_ids"][0][idx].item()) for idx in top_indices]
                
                # Store results
                attention_scores_per_token[token_idx] = top_tokens

        return attention_scores_per_token

# Example usage
analyzer = AttentionAnalyzer(model_name="meta-llama/Llama-3.2-1B")
input_text = "The llama wandered through the mountains."
most_attended_tokens = analyzer.get_most_attended_tokens(input_text)
for idx, tokens in most_attended_tokens.items():
    print(f"Token {idx}: Most attended tokens: {tokens}")
