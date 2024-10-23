import torch

class SlidingWindowManager:
    def __init__(self, window_size, device, max_tokens_part1=128, max_tokens_part2=128, max_tokens_part3=64, max_tokens_part4=200):
        """
        Initialize the Sliding Window Manager.
        :param window_size: Total number of tokens in the sliding window.
        :param device: torch.device (cpu or cuda)
        :param max_tokens_part1: Maximum tokens for RAG and HyDE Data.
        :param max_tokens_part2: Maximum tokens for Prompted Data.
        :param max_tokens_part3: Maximum tokens for Primary and Secondary Queries.
        :param max_tokens_part4: Maximum tokens for Response.
        """
        self.window_size = window_size
        self.device = device
        self.max_tokens_part1 = max_tokens_part1
        self.max_tokens_part2 = max_tokens_part2
        self.max_tokens_part3 = max_tokens_part3
        self.max_tokens_part4 = max_tokens_part4

        self.reset_window()

    def reset_window(self):
        """Reset the sliding window to its initial state."""
        self.parts = {
            'part1': [],  # RAG and HyDE Data
            'part2': [],  # Prompted Data
            'part3': [],  # Primary and Secondary Queries
            'part4': [],  # Response
        }
    
    def tokenize_text(self, text, tokenizer, max_tokens):
        """
        Tokenizes and truncates the input text.
        :param text: Input string.
        :param tokenizer: Tokenizer object.
        :param max_tokens: Maximum tokens to use for truncation.
        :return: List of tokenized ids.
        """
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return tokens[:max_tokens]

    def parse_prompt(self, prompt, tokenizer):
        """
        Parse the prompt to identify primary and secondary queries.
        :param prompt: The full input prompt string.
        :param tokenizer: The tokenizer used for encoding.
        :return: List of tokens for the queries.
        """
        sentences = prompt.split('.')
        queries = [s.strip() for s in sentences if '?' in s]
        primary_query = queries[0] if queries else ""
        secondary_queries = queries[1:] if len(queries) > 1 else []

        # Combine primary and secondary queries
        combined_queries = primary_query + " " + " ".join(secondary_queries)
        tokens = self.tokenize_text(combined_queries, tokenizer, self.max_tokens_part3)
        return tokens

    def fill_parts(self, tokens_part1, tokens_part3):
        """
        Fill the sliding window parts with tokens.
        :param tokens_part1: RAG and HyDE data tokens.
        :param tokens_part3: Primary and secondary query tokens.
        """
        # Fill the relevant parts
        self.parts['part1'] = tokens_part1[:self.max_tokens_part1]
        self.parts['part3'] = tokens_part3[:self.max_tokens_part3]
        
        # Ensure part2 and part4 are initialized
        self.parts['part2'] = []  # Example context or prompted data
        self.parts['part4'] = []  # Response window starts empty

    def get_combined_query(self):
        """
        Combine all parts into a single query.
        :return: Combined token list.
        """
        combined = []
        for part in self.parts.values():
            combined.extend(part)
        
        return combined[:self.window_size]  # Ensure total length doesn't exceed window size

    def update_window_attention(self, attention_scores_part1, attention_scores_part2):
        """
        Dynamically update the sliding window based on attention scores.
        :param attention_scores_part1: Attention scores for part1.
        :param attention_scores_part2: Attention scores for part2.
        """
        concentration_part1 = self.calculate_attention_concentration(attention_scores_part1)
        concentration_part2 = self.calculate_attention_concentration(attention_scores_part2)
        
        # Threshold to decide whether to flush or shift windows
        threshold = 0.5

        if concentration_part1 > threshold:
            self.flush_window('part1')
        elif concentration_part2 > threshold:
            self.shift_response_window()

    def calculate_attention_concentration(self, attention_scores):
        """
        Calculate the attention concentration towards the end of a sequence.
        :param attention_scores: Attention scores tensor for a specific part.
        :return: Attention concentration score.
        """
        if attention_scores.numel() == 0:
            return 0
        
        # Average the attention at the last position
        return attention_scores[:, -1].mean().item()

    def flush_window(self, part_key):
        """
        Flush half of the tokens from the specified part of the window.
        :param part_key: The key for the part to flush.
        """
        half_length = len(self.parts[part_key]) // 2
        self.parts[part_key] = self.parts[part_key][half_length:]
        print(f"Sliding Window: Flushed half of {part_key}.")

    def shift_response_window(self):
        """
        Push the response window by one token, simulating a sliding window.
        """
        if len(self.parts['part4']) > 0:
            self.parts['part4'] = self.parts['part4'][1:]
            print("Sliding Window: Shifted the Response Window by one token.")

    def add_response_tokens(self, response_tokens):
        """
        Add new tokens to the response part of the window.
        :param response_tokens: List of tokenized response tokens.
        """
        self.parts['part4'].extend(response_tokens)
        if len(self.parts['part4']) > self.max_tokens_part4:
            self.parts['part4'] = self.parts['part4'][-self.max_tokens_part4:]  # Keep only the latest tokens
