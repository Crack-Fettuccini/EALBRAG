from sliding_window import SlidingWindowManager
from model import TokenGeneratorWithAttention

class PromptManager:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", window_size=1024):
        # Initialize the model and the sliding window manager
        self.token_generator = TokenGeneratorWithAttention(model_name=model_name, window_size=window_size)
        self.sliding_window_manager = SlidingWindowManager(model_name=model_name, window_size=window_size)

    def process_prompt(self, input_prompt, max_tokens=10):
        """
        Process the input prompt by generating tokens until the end token is reached.
        
        Parameters:
            input_prompt (str): The initial prompt to begin processing.
            max_tokens (int): Maximum tokens to generate in each model call.
        
        Returns:
            str: Generated text as a single string until the end token is encountered.
        """
        # Initialize tracking variables
        output_tokens = []         # Collect output tokens
        current_prompt = input_prompt

        # Loop until end token is reached
        while True:
            # Check if the last token in output tokens is the end token
            if output_tokens and output_tokens[-1] == self.token_generator.tokenizer.eos_token_id:
                break  # End token found, exit loop

            # Pass the current prompt to the sliding window manager and update current prompt
            combined_prompt = self.sliding_window_manager.process_input(input_prompt, current_prompt)
            current_prompt = combined_prompt

            # Generate the next token using the model
            generated_text = self.token_generator.generate_tokens(current_prompt, max_tokens=1)
            
            # Tokenize the generated text and retrieve the token id
            generated_token_ids = self.token_generator.tokenizer.encode(generated_text, add_special_tokens=False)
            output_tokens.extend(generated_token_ids)  # Append token to output tokens

            # Update current prompt with the new tokens generated
            current_prompt += generated_text

        # Decode the final output tokens to return the generated text
        final_output = self.token_generator.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return final_output
