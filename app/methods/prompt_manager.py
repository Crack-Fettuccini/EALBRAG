# prompt_manager.py
from sliding_window import SlidingWindowManager

class PromptManager:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", window_size=50):
        self.sliding_window_manager = SlidingWindowManager(model_name=model_name, window_size=window_size)

    def process_prompt(self, input_text, max_tokens=10):
        """
        Processes the input prompt, checking for end tokens and managing the sliding window.
        
        Parameters:
            input_text (str): The input prompt to process.
            max_tokens (int): Maximum number of tokens to generate.
        
        Returns:
            str: The final generated output once the end token is reached.
        """
        while True:
            generated_output = self.sliding_window_manager.generate_tokens(input_text, max_tokens=max_tokens)
            if generated_output.strip() and generated_output[-len(self.sliding_window_manager.tokenizer.eos_token):] == self.sliding_window_manager.tokenizer.eos_token:
                return generated_output
            else:
                input_text += generated_output
                print(f"Continuing with input: {input_text}")
"""
# Example usage
if __name__ == "__main__":
    prompt_manager = PromptManager()
    initial_input = "The llama wandered through the mountains."
    final_output = prompt_manager.process_prompt(initial_input, max_tokens=5)
    print("Final Generated Output:", final_output)
"""