import torch
from HyDE import HyDE
from HyPE import HyPE
from query_identifier import QueryIdentifier

class SlidingWindowManager:
    def __init__(self, window_size, device, model_hype, model_hyde, tokenizer_hype, tokenizer_hyde, max_tokens_part1=128, max_tokens_part2=128, max_tokens_part3=64, max_tokens_part4=200):
        """
        Initialize the Sliding Window Manager.
        :param window_size: Total number of tokens in the sliding window.
        :param device: torch.device (cpu or cuda)
        :param model_hype: Pretrained model for HyPE.
        :param model_hyde: Pretrained model for HyDE.
        :param tokenizer_hype: Tokenizer corresponding to the HyPE model.
        :param tokenizer_hyde: Tokenizer corresponding to the HyDE model.
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
        
        self.hyde = HyDE(model_hyde, tokenizer_hyde, device)
        self.hype = HyPE(model_hype, tokenizer_hype, device)
        self.query_identifier = QueryIdentifier()
        
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
        self.query_identifier.process_corpus(prompt)  # Process the prompt to identify queries

        sentences = self.query_identifier.split_into_sentences(prompt)
        possible_queries = self.query_identifier.detect_possible_queries(sentences)
        
        if not possible_queries:
            return []  # Return empty if no queries found

        main_content, queries = self.query_identifier.identify_content_and_queries(sentences, possible_queries)
        primary_queries, secondary_queries = self.query_identifier.categorize_queries(main_content, queries)

        # Combine primary and secondary queries
        combined_queries = primary_queries + secondary_queries
        combined_query_text = " ".join(combined_queries)
        tokens = self.tokenize_text(combined_query_text, tokenizer, self.max_tokens_part3)
        return tokens

    def fill_parts_with_hyde_and_hype(self, query, conversation_history, retrieved_docs):
        """
        Fill part1 using HyDE and HyPE generated data.
        :param query: Query string to generate HyDE hypothetical documents.
        :param conversation_history: List of conversation strings for HyPE.
        :param retrieved_docs: List of retrieved documents for HyPE ground truth generation.
        """
        # Generate hypothetical documents using HyDE
        hyde_results = self.hyde.generate_and_return_token_ids(query)
        hyde_tokens = hyde_results[0][1]  # Extract token ids from the first result
        hyde_tokens = hyde_tokens[:self.max_tokens_part1]

        # Generate hypothetical ground truths using HyPE
        hype_ground_truths = self.hype.generate_hypothetical_ground_truths(conversation_history, retrieved_docs)
        if hype_ground_truths:
            hype_tokens = self.tokenize_text(hype_ground_truths[0], self.hype.tokenizer, self.max_tokens_part1)
            # Combine HyDE and HyPE tokens for part1
            combined_tokens_part1 = hyde_tokens[:self.max_tokens_part1 // 2] + hype_tokens[:self.max_tokens_part1 // 2]
        else:
            combined_tokens_part1 = hyde_tokens
        
        # Fill the part1 with the combined tokens
        self.parts['part1'] = combined_tokens_part1
    
    def fill_parts(self, tokens_part3):
        """
        Fill the sliding window parts with tokens.
        :param tokens_part3: Primary and secondary query tokens.
        """
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
