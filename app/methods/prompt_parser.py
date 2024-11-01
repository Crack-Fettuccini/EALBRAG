from sliding_window import SlidingWindowManager
from model import TokenGeneratorWithAttention

class PromptManager:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", window_size=1024):
        """
        Initialize PromptManager with a token generator and sliding window manager.

        Parameters:
            model_name (str): Name of the language model.
            window_size (int): Maximum number of tokens for the sliding window.
        """
        # Initialize the token generator with attention and sliding window manager
        self.token_generator = TokenGeneratorWithAttention(model_name=model_name, window_size=window_size)
        self.sliding_window_manager = SlidingWindowManager(model_name=model_name, window_size=window_size)

    def process_prompt(self, input_prompt, max_tokens=50):
        """
        Process the input prompt by generating tokens until the end token is reached.

        Parameters:
            input_prompt (str): Initial text prompt to start generating from.
            max_tokens (int): Maximum tokens to generate per call.

        Returns:
            str: Final generated text until the end token.
        """
        # Tracking variables for token generation and sliding window prompt updates
        output_tokens = []     # Collects generated token IDs
        current_prompt = input_prompt

        while len(output_tokens) < max_tokens:
            # If an end-of-sequence token has been generated, stop the loop
            if output_tokens and output_tokens[-1] == self.token_generator.tokenizer.eos_token_id:
                break  # Stop if end-of-sequence token is reached

            # Update current prompt using sliding window manager to manage prompt length
            combined_prompt = self.sliding_window_manager.process_input(input_prompt, current_prompt)
            current_prompt = combined_prompt

            # Generate the next token and add it to the output tokens
            generated_text = self.token_generator.generate_tokens(current_prompt, max_length=1)
            generated_token_ids = self.token_generator.tokenizer.encode(generated_text, add_special_tokens=False)
            output_tokens.extend(generated_token_ids)

            # Append new tokens to the current prompt to maintain context
            current_prompt += generated_text

        # Decode collected tokens to form the final output response
        final_output = self.token_generator.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return final_output
