import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TokenGeneratorWithAttention:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", window_size=50, attention_threshold=0.05):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        
        # Configuration parameters
        self.window_size = window_size
        self.attention_threshold = attention_threshold

    def generate_tokens_with_attention(self, input_text, max_tokens=10):
        """
        Generates tokens one by one, checking attention on the last tokens of the input.
        
        Parameters:
            input_text (str): The input text to analyze and generate tokens from.
            max_tokens (int): Maximum number of tokens to generate.
        
        Returns:
            tuple: A list of attention scores and the generated text.
        """
        generated_tokens = []
        attention_scores_list = []
        
        # Tokenize the input text
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        input_length = input_ids.shape[1]

        for _ in range(max_tokens):
            # Ensure we only consider the last `window_size` tokens
            current_window = input_ids[:, max(0, input_length - self.window_size):]
            
            # Generate one token
            with torch.no_grad():
                outputs = self.model(current_window, return_dict=True, output_attentions=True)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)  # Get the token with the highest probability

            # Append the generated token to the list
            generated_tokens.append(next_token_id.item())
            
            # Prepare for the next iteration by appending the new token
            input_ids = torch.cat([current_window, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            input_length += 1
            
            # Check attention scores for the last tokens
            last_attention_scores = outputs.attentions[-1][:, :, -self.window_size:, -self.window_size:]  # Last layer's attention
            avg_attention_scores = last_attention_scores.mean(dim=1).squeeze()  # Average over heads
            
            # Get the attention score for the newly generated token
            last_token_attention = avg_attention_scores[-1].item()  # Attention on the last token
            attention_scores_list.append(avg_attention_scores.tolist())  # Collect attention scores
            
            print(f"Attention on last token: {last_token_attention:.4f}")

            if last_token_attention < self.attention_threshold:
                print("Insufficient attention on the last token, stopping generation.")
                break  # Stop if attention is below the threshold

        # Decode the generated tokens to strings
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return attention_scores_list, generated_text
"""
# Example usage
token_generator = TokenGeneratorWithAttention(model_name="meta-llama/Llama-3.2-1B")
input_text = "The llama wandered through the mountains."
attention_scores, generated_output = token_generator.generate_tokens_with_attention(input_text, max_tokens=5)
print("Attention Scores per Token:", attention_scores)
print("Generated Output:", generated_output)
"""