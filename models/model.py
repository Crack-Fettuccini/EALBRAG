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

# Function to track token importance based on attention scores across all heads
def track_attention_per_head(attentions, input_ids, threshold=attention_threshold):
    token_attention_info = {}
    
    # Iterate over each layer and head, collecting attention scores for each token
    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn shape: (batch_size, num_heads, seq_len, seq_len)
        avg_attn_per_token = layer_attn.mean(dim=1)  # Average over heads
        important_tokens = (avg_attn_per_token > threshold).nonzero(as_tuple=True)[1]

        # Initialize dictionary for layer's tokens
        token_attention_info[layer_idx] = {}

        for head_idx in range(layer_attn.shape[1]):
            # Attention for current head
            head_attn = layer_attn[:, head_idx, :, :]
            for token_idx in range(head_attn.shape[-1]):
                token_id = input_ids[0, token_idx].item()
                token = tokenizer.decode([token_id])

                # Store attention scores across heads
                if token_id not in token_attention_info[layer_idx]:
                    token_attention_info[layer_idx][token_id] = {
                        "token": token,
                        "attention_scores": []
                    }

                # Append current head's attention score for the token
                token_attention_info[layer_idx][token_id]["attention_scores"].append(
                    head_attn[0, token_idx].tolist()  # Per-head score for token
                )

    return token_attention_info

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

        # Track detailed per-head attention for each token
        token_attention_info = track_attention_per_head(attentions, generated)
        memory.extend([tok for layer_info in token_attention_info.values() 
                       for tok in layer_info.keys()])  # Update memory with important tokens

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
    return response, token_attention_info

# Usage example
prompt = "Explain the theory of relativity in simple terms."
response, attention_data = generate_response(prompt, max_length=50, verbose=True)
print("\nFinal Response:", response)

# Print attention information for a specific token in the first layer as an example
layer_idx = 0  # First layer
token_id_example = list(attention_data[layer_idx].keys())[0]  # Example token id from layer
print(f"\nAttention details for token '{attention_data[layer_idx][token_id_example]['token']}' in Layer {layer_idx}:\n",
      attention_data[layer_idx][token_id_example]["attention_scores"])
