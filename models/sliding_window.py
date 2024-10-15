# models/sliding_window.py
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
        """
        Reset the sliding window to its initial state.
        """
        self.part1 = []  # RAG and HyDE Data
        self.part2 = []  # Prompted Data
        self.part3 = []  # Primary and Secondary Queries
        self.part4 = []  # Response
    
    def parse_prompt(self, prompt, tokenizer, max_tokens_part3):
        """
        Parse the prompt to identify primary and secondary queries.
        :param prompt: The full input prompt string.
        :param tokenizer: The tokenizer used for encoding.
        :param max_tokens_part3: Maximum tokens allocated for part 3.
        :return: List of tokens for part 3.
        """
        # For simplicity, assume primary query is the main question,
        # and secondary queries are additional inquiries.
        # This can be enhanced using NLP techniques for better parsing.

        # Example implementation:
        # Split the prompt into sentences and identify queries.

        sentences = prompt.split('.')
        queries = [s.strip() for s in sentences if '?' in s]
        primary_query = queries[0] if queries else ""
        secondary_queries = queries[1:] if len(queries) > 1 else []

        # Combine primary and secondary queries, limited by max_tokens_part3
        combined_queries = primary_query + " " + " ".join(secondary_queries)
        tokens = tokenizer.encode(combined_queries, add_special_tokens=False)
        tokens = tokens[:max_tokens_part3]  # Truncate if necessary

        return tokens

    def fill_parts(self, tokens_part3, tokens_part1):
        """
        Fill the sliding window parts with tokens.
        :param tokens_part3: Primary and secondary query tokens.
        :param tokens_part1: RAG and HyDE data tokens.
        """
        # Fill Part 3
        self.part3 = tokens_part3

        # Fill Part 1 with RAG and HyDE data
        self.part1 = tokens_part1[:self.max_tokens_part1]

        # Fill Part 2 with prompted data (can be extended as needed)
        self.part2 = []  # Initialize as empty or fill with additional context if available

        # Initialize Part 4 (Response) as empty
        self.part4 = []

    def get_combined_query(self):
        """
        Combine all parts into a single query.
        :return: Combined token list.
        """
        combined = self.part1 + self.part2 + self.part3 + self.part4
        return combined[:self.window_size]  # Ensure it doesn't exceed window size

    def update_window_attention(self, attention_scores_part1, attention_scores_part2):
        """
        Update the sliding window based on attention scores.
        :param attention_scores_part1: Attention scores for part1.
        :param attention_scores_part2: Attention scores for part2.
        """
        # Calculate attention concentration towards the end of parts
        concentration_part1 = attention_scores_part1[:, -1].mean().item() if attention_scores_part1.numel() > 0 else 0
        concentration_part2 = attention_scores_part2[:, -1].mean().item() if attention_scores_part2.numel() > 0 else 0

        # Define a threshold for concentration (e.g., 0.5)
        threshold = 0.5

        if concentration_part1 > threshold:
            # Flush half of part1
            half_length = len(self.part1) // 2
            self.part1 = self.part1[half_length:]
            print("Sliding Window: Flushing half of Part 1 (RAG and HyDE Data)")
        elif concentration_part2 > threshold:
            # Push response window by one token
            if len(self.part4) > 0:
                self.part4 = self.part4[1:]
                print("Sliding Window: Pushing Response Window by one token")
