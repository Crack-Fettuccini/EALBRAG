import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.memory_attention import MemoryAttention  # Ensure this path is correct
from collections import deque

class TokenGeneratorWithAttention:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8", max_window_size=2048, attention_threshold=0.1, memory_size=1000):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)

        # Memory attention parameters
        embed_dim = self.model.config.hidden_size
        num_heads = self.model.config.num_attention_heads
        lstm_hidden_size = 256
        num_lstm_layers = 1
        dropout = 0.1

        # Initialize MemoryAttention
        self.memory_attention = MemoryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            memory_size=memory_size,
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout
        )

        # Initialize other parameters
        self.max_window_size = max_window_size
        self.attention_threshold = attention_threshold
        self.memory = deque(maxlen=memory_size)  # Memory storage
        self.last_output_cache = None  # Cache for last generated result

    def track_attention(self, attentions, input_ids):
        """Identify and track important tokens based on attention scores."""
        important_tokens = set()
        for layer_attention in attentions:
            avg_attention = layer_attention.mean(dim=1)  # Average across heads
            # Track tokens that exceed the attention threshold
            important_tokens.update((avg_attention > self.attention_threshold).nonzero(as_tuple=True)[1].tolist())
        return list(important_tokens)

    def generate_tokens(self, input_prompt, max_length=100, verbose=False):
        """Generate tokens with attention-based memory and sliding window."""
        # Check if input matches the last prompt; if so, return the cached output
        if self.last_output_cache and self.last_output_cache["input_prompt"] == input_prompt:
            return self.last_output_cache["response"]

        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids
        attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        generated = input_ids.clone()  # Clone for sliding window

        for _ in range(max_length):
            # Forward pass to get logits and attention scores
            outputs = self.model(input_ids=generated, attention_mask=attention_mask)
            logits = outputs.logits
            attentions = outputs.attentions

            # Track and store important tokens based on attention scores
            important_tokens = self.track_attention(attentions, generated)
            # Update memory with new important tokens, ensuring no duplicates
            self.memory.extend(tok for tok in important_tokens if tok not in self.memory)

            # Generate next token
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token_id], dim=1)

            # Manage sliding window
            if generated.shape[1] > self.max_window_size:
                generated = generated[:, -self.max_window_size:]

            attention_mask = torch.ones(generated.shape, device=generated.device)

            # Optional verbose logging of each generated token
            if verbose:
                next_token = self.tokenizer.decode(next_token_id)
                print(f"Generated Token: {next_token}")

            # Stop if end-of-sequence token is generated
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        # Decode generated tokens
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        # Cache the output for future reference
        self.last_output_cache = {"input_prompt": input_prompt, "response": response}
        return response