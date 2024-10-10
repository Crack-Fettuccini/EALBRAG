# models/sliding_window.py
import torch
import torch.nn as nn
import spacy

class SlidingWindowManager:
    def __init__(self, window_size, device):
        """
        Initialize the Sliding Window Manager.
        :param window_size: Total number of tokens in the sliding window.
        :param device: torch.device (cpu or cuda)
        """
        self.window_size = window_size
        self.device = device
        self.reset_window()
        self.nlp = spacy.load("en_core_web_sm")

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
        Parse the prompt to identify primary and secondary queries using spaCy.
        :param prompt: The full input prompt string.
        :param tokenizer: The tokenizer used for encoding.
        :param max_tokens_part3: Maximum tokens allocated for part 3.
        :return: List of tokens for part 3.
        """
        doc = self.nlp(prompt)
        queries = []
        primary_query = ""
        secondary_queries = []

        # Identify sentences with interrogative clauses as queries
        for sent in doc.sents:
            if sent[-1].text == '?':
                if not primary_query:
                    primary_query = sent.text
                else:
                    secondary_queries.append(sent.text)

        # Combine primary and secondary queries
        if primary_query:
            queries.append(primary_query)
        queries.extend(secondary_queries)

        # Combine and encode
        combined_queries = " ".join(queries)
        tokens = tokenizer.encode(combined_queries, add_special_tokens=False)
        tokens = tokens[:max_tokens_part3]  # Truncate if necessary

        return tokens

    def fill_parts(self, tokens, rag_tokens, max_tokens_part1, max_tokens_part2, max_tokens_part4):
        """
        Fill the sliding window parts with tokens.
        :param tokens: Primary and secondary query tokens (part 3).
        :param rag_tokens: RAG and HyDE data tokens (part 1).
        :param max_tokens_part1: Maximum tokens for part 1.
        :param max_tokens_part2: Maximum tokens for part 2.
        :param max_tokens_part4: Maximum tokens for part 4.
        """
        # Fill Part 3
        self.part3 = tokens

        # Fill Part 2 with prompted data (for simplicity, assume empty initially)
        self.part2 = []

        # Fill Part 1 with RAG and HyDE data
        self.part1 = rag_tokens[:max_tokens_part1]

        # Initialize Part 4 (Response) as empty
        self.part4 = []

    def update_window_attention(self, attention_scores_part1, attention_scores_part2):
        """
        Update the sliding window based on attention scores.
        :param attention_scores_part1: Attention scores for part 1.
        :param attention_scores_part2: Attention scores for part 2.
        """
        # Determine if attention is concentrated towards the end of part1 or part2
        # For simplicity, assume attention_scores are averaged across heads and tokens

        # Calculate attention concentration
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

    def add_to_part1(self, new_tokens, max_tokens_part1):
        """
        Add new tokens to part1 while respecting the maximum token limit.
        :param new_tokens: List of new tokens to add.
        :param max_tokens_part1: Maximum tokens allowed for part1.
        """
        self.part1 += new_tokens
        if len(self.part1) > max_tokens_part1:
            self.part1 = self.part1[-max_tokens_part1:]
            print("Sliding Window: Part 1 exceeded maximum tokens, truncating.")

    def add_to_part2(self, new_tokens, max_tokens_part2):
        """
        Add new tokens to part2 while respecting the maximum token limit.
        :param new_tokens: List of new tokens to add.
        :param max_tokens_part2: Maximum tokens allowed for part2.
        """
        self.part2 += new_tokens
        if len(self.part2) > max_tokens_part2:
            self.part2 = self.part2[-max_tokens_part2:]
            print("Sliding Window: Part 2 exceeded maximum tokens, truncating.")

    def add_to_part4(self, new_tokens, max_tokens_part4):
        """
        Add new tokens to part4 while respecting the maximum token limit.
        :param new_tokens: List of new tokens to add.
        :param max_tokens_part4: Maximum tokens allowed for part4.
        """
        self.part4 += new_tokens
        if len(self.part4) > max_tokens_part4:
            self.part4 = self.part4[-max_tokens_part4:]
            print("Sliding Window: Part 4 exceeded maximum tokens, truncating.")

    def get_combined_query(self):
        """
        Combine all parts into a single query.
        :return: Combined token list.
        """
        combined = self.part1 + self.part2 + self.part3 + self.part4
        return combined[:self.window_size]  # Ensure it doesn't exceed window size
