# generate_with_memory.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.memory_attention import MemoryAttention  # Ensure this path is correct
from collections import deque

# Model and Tokenizer Setup
model_name = "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)

# MemoryAttention initialization parameters
embed_dim = model.config.hidden_size  # Based on the model's configuration
num_heads = model.config.num_attention_heads
memory_size = 1000
lstm_hidden_size = 256
num_lstm_layers = 1
dropout = 0.1

# Initialize MemoryAttention module
memory_attention = MemoryAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    memory_size=memory_size,
    lstm_hidden_size=lstm_hidden_size,
    num_lstm_layers=num_lstm_layers,
    dropout=dropout
)

# Define parameters for generation and memory
max_window_size = 2048  # Sliding window size based on model capacity
attention_threshold = 0.1  # Threshold for token importance
memory = deque(maxlen=memory_size)  # Memory store for top tokens

# Track token importance using attention weights
def track_attention(attentions, input_ids, threshold=attention_threshold):
    important_tokens = []
    for layer_attention in attentions:
        avg_attention = layer_attention.mean(dim=1)  # Average across heads
        important_tokens += (avg_attention > threshold).nonzero(as_tuple=True)[1].tolist()
    return important_tokens

# Generate response with memory-based attention tracking
def generate_response(prompt, max_length=100, verbose=False):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
    generated = input_ids.clone()  # Clone for sliding window

    for _ in range(max_length):
        # Forward pass through the model to get logits and attention scores
        outputs = model(input_ids=generated, attention_mask=attention_mask)
        logits = outputs.logits
        attentions = outputs.attentions

        # Track important tokens from attention data
        important_tokens = track_attention(attentions, generated)
        memory.extend(important_tokens)

        # Generate next token
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token_id], dim=1)

        if generated.shape[1] > max_window_size:
            generated = generated[:, -max_window_size:]

        attention_mask = torch.ones(generated.shape, device=generated.device)

        if verbose:
            next_token = tokenizer.decode(next_token_id)
            print(f"Generated Token: {next_token}")

        # Check for end of sequence
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Decode final response
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    return response, important_tokens

# Example usage
prompt = "Explain the concept of gravity in simple terms."
response, attention_data = generate_response(prompt, max_length=50, verbose=True)
print("\nFinal Response:", response)
