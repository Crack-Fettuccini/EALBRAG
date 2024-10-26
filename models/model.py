import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import deque

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)

# Define parameters
max_window_size = 2048  # Sliding window size (adapt based on model capacity)
memory_size = 256       # Maximum tokens in LSTM-based memory
attention_threshold = 0.1  # Threshold for important tokens

# Initialize LSTM-based memory (a deque to store top important tokens)
memory = deque(maxlen=memory_size)

# Function to track token importance based on attention scores
def track_attention(attentions, input_ids, threshold=attention_threshold):
    token_importance = []
    for layer_attn in attentions:
        avg_attn = layer_attn.mean(dim=1)  # Mean over all heads
        important_tokens = (avg_attn > threshold).nonzero(as_tuple=True)[1]
        token_importance.extend(input_ids[important_tokens].tolist())
    return token_importance

# Initialize context and generate tokens one at a time
def generate_response(prompt, max_length=100, verbose=False):
    # Tokenize input prompt and convert to tensor
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
    generated = input_ids.clone()  # Clone for sliding window

    # Generate tokens one at a time
    for _ in range(max_length):
        # Forward pass to obtain logits and attention
        outputs = model(input_ids=generated, attention_mask=attention_mask)
        logits = outputs.logits
        attentions = outputs.attentions  # Shape: (num_layers, batch, num_heads, seq_len, seq_len)

        # Token importance tracking
        important_tokens = track_attention(attentions, generated)
        memory.extend(important_tokens)  # Update memory with important tokens

        # Sample the next token
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # Update context with new token and shift window if max length exceeded
        generated = torch.cat([generated, next_token_id], dim=1)
        if generated.shape[1] > max_window_size:
            generated = generated[:, -max_window_size:]
        
        # Update attention mask
        attention_mask = torch.ones(generated.shape, device=generated.device)

        # Decode the generated token and print if verbose
        if verbose:
            next_token = tokenizer.decode(next_token_id)
            print(f"Generated Token: {next_token}")
        
        # End generation if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Decode final generated response
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    return response

# Usage example
prompt = "Explain the theory of relativity in simple terms."
response = generate_response(prompt, max_length=50, verbose=True)
print("\nFinal Response:", response)

# Print top tokens in memory
print("\nImportant Tokens in Memory:", [tokenizer.decode(tok) for tok in list(memory)])
