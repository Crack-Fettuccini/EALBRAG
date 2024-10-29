from sliding_window import SlidingWindowManager
from model import TokenGeneratorWithAttention

class PromptParser:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", window_size=1024, attention_threshold=0.05):
        # Initialize the SlidingWindowManager and TokenGeneratorWithAttention
        self.sliding_window_manager = SlidingWindowManager(model_name=model_name, window_size=window_size)
        self.token_generator = TokenGeneratorWithAttention(model_name=model_name, window_size=window_size, attention_threshold=attention_threshold)

    def process_prompt(self, input_text, max_tokens=10):
        """
        Processes the input prompt, checking for end tokens and managing the sliding window and attention-based generation.
        
        Parameters:
            input_text (str): The input prompt to process.
            max_tokens (int): Maximum number of tokens to generate.
        
        Returns:
            str: The final generated output once the end token is reached or attention is insufficient.
        """
        # Initialize the sliding window with the input text
        self.sliding_window_manager.reset_window()
        self.sliding_window_manager.fill_parts(self.sliding_window_manager.tokenize_text(input_text, self.token_generator.tokenizer, max_tokens))
        
        final_output = ""
        
        while True:
            # Get combined query with the current window
            combined_query_tokens = self.sliding_window_manager.get_combined_query()
            combined_query_text = self.token_generator.tokenizer.decode(combined_query_tokens, skip_special_tokens=True)
            
            # Generate tokens with attention checks
            attention_scores, generated_text = self.token_generator.generate_tokens_with_attention(combined_query_text, max_tokens=max_tokens)
            final_output += generated_text
            
            # Add generated tokens to response part of the sliding window
            generated_token_ids = self.token_generator.tokenizer.encode(generated_text, add_special_tokens=False)
            self.sliding_window_manager.add_response_tokens(generated_token_ids)
            
            # Check for end token or insufficient attention
            if generated_text.strip() and generated_text[-len(self.token_generator.tokenizer.eos_token):] == self.token_generator.tokenizer.eos_token:
                break
            elif any(score < self.token_generator.attention_threshold for score in attention_scores):
                print("Attention fell below threshold; stopping generation.")
                break
            else:
                print(f"Continuing with next window segment.")

        return final_output
