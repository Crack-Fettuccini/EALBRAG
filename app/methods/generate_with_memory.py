import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.memory_attention import MemoryAttention  # Ensure this path is correct
from collections import deque
from HyPE import HyPE

class TokenGeneratorWithAttention:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8", max_window_size=2048, attention_threshold=0.1, memory_size=1000):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_attentions=True, 
            torch_dtype=torch.float16  # Use float16 for efficiency if supported
        ).cuda()  # Use GPU if available

        # Memory attention parameters
        embed_dim = self.model.config.hidden_size
        num_heads = self.model.config.num_attention_heads
        lstm_hidden_size = 256
        num_lstm_layers = 1
        dropout = 0.1

        # Initialize MemoryAttention for long-term memory
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
        self.memory = deque(maxlen=memory_size)  # Store important tokens in memory
        self.hype = HyPE()  # Hypothetical grounding model

    def track_attention(self, attentions):
        """
        Identify important tokens based on attention scores.
        
        :param attentions: Attention scores from the model's forward pass.
        :return: List of important token IDs.
        """
        important_tokens = set()
        for layer_attention in attentions:
            avg_attention = layer_attention.mean(dim=1)  # Average attention across heads
            important_tokens.update(
                (avg_attention > self.attention_threshold).nonzero(as_tuple=True)[1].tolist()
            )
        return list(important_tokens)

    def generate_tokens(self, input_prompt, max_length=100, verbose=False):
        """
        Generate tokens using a sliding window mechanism and track important tokens.
        
        :param input_prompt: Input text prompt.
        :param max_length: Maximum number of tokens to generate.
        :param verbose: If True, print each generated token.
        :return: Generated response and hypothetical ground truths.
        """
        # Encode the input prompt
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        generated = input_ids.clone()

        for _ in range(max_length):
            # Forward pass through the model
            outputs = self.model(input_ids=generated, attention_mask=attention_mask)
            logits, attentions = outputs.logits, outputs.attentions

            # Track important tokens based on attention scores
            important_tokens = self.track_attention(attentions)
            self.memory.extend(tok for tok in important_tokens if tok not in self.memory)

            # Generate next token
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token_id], dim=1)

            # Manage the sliding window
            if generated.size(1) > self.max_window_size:
                generated = generated[:, -self.max_window_size:]

            attention_mask = torch.ones_like(generated, device=generated.device)

            if verbose:
                print(f"Generated Token: {self.tokenizer.decode(next_token_id)}")

            # Stop generation if the EOS token is encountered
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        # Decode the generated response
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        # Use HyPE to generate hypothetical ground truths
        ground_truths = self.hype.generate_hypothetical_ground_truths([input_prompt], [])

        return response, ground_truths
