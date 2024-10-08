import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import sys
import matplotlib.pyplot as plt

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the tokenizer and model
model_name = "NousResearch/Llama-3.2-1B"  # Replace with your model name or path

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

print("Loading model...")
model = LlamaForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    low_cpu_mem_usage=True  # Helps with large models on limited CPU memory
)
#model.to(device)
model.eval()

# 2. Function to run inference and collect attention
def run_inference(input_text, max_new_tokens=50):
    """
    Runs the model on the input_text and collects attention scores.

    Args:
        input_text (str): The input prompt for the model.
        max_new_tokens (int): The maximum number of tokens to generate.

    Returns:
        inputs (dict): Tokenized input.
        outputs (GenerationOutput): Model's generation output containing attentions.
    """
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate output with attentions
    print("Running inference...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print("\nGenerated Text:")
    print(generated_text)

    return inputs, outputs

import torch
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_visualize_attention(inputs, outputs, keywords=[], method="z-score", z_threshold=2, iqr_factor=1.5):
    """
    Analyzes the collected attention scores to determine token contributions
    and visualizes the most attended input words, ignoring specified outliers and keywords.
    Outliers can be filtered based on z-scores or interquartile range (IQR).

    Args:
        inputs (dict): Tokenized input.
        outputs (GenerationOutput): Model's generation output containing attentions.
        keywords (list, optional): List of tokens to ignore in the analysis. Defaults to [].
        method (str, optional): Method for outlier detection ('z-score' or 'quartile'). Defaults to 'z-score'.
        z_threshold (float, optional): Z-score threshold for filtering (only used if method='z-score'). Defaults to 2.
        iqr_factor (float, optional): Factor for IQR-based filtering (only used if method='quartile'). Defaults to 1.5.
    """
    # Extract input and generated tokens
    input_ids = inputs["input_ids"][0]
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Clean input tokens by removing special characters (e.g., 'Ġ')
    input_tokens_clean = [token.replace('Ġ', ' ').strip() for token in input_tokens]

    # Extract generated tokens, excluding input tokens
    generated_ids = outputs.sequences[0][input_ids.shape[0]:]  
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

    # Handle attentions (ensure 'output_attentions=True' is set when generating)
    if not outputs.attentions:
        print("No attentions were returned. Ensure that 'output_attentions=True' is set.")
        return

    # Stack attentions from all layers: Shape (num_layers, batch_size, num_heads, seq_length, seq_length)
    attentions = torch.stack(outputs.attentions[0]).detach().cpu()  # Shape: (num_layers, batch, num_heads, seq, seq)

    num_layers, batch_size, num_heads, seq_length, _ = attentions.shape
    print(f"\nNumber of layers with attention scores: {num_layers}")
    print(f"Number of attention heads per layer: {num_heads}")
    print(f"Total sequence length (input + generated): {seq_length}")

    # Initialize structures to hold aggregated attention scores and counts
    aggregated_scores = torch.zeros(len(input_tokens_clean))
    passes = torch.zeros(len(input_tokens_clean))

    # Average attentions over all layers and heads
    avg_attn = attentions.mean(dim=2).mean(dim=0)  # Shape: (batch_size, seq_length, seq_length)
    avg_attn = avg_attn[0]  # Since batch_size=1, extract the attention matrix for the first example

    # Focus on attentions from generated tokens to input tokens
    input_len = input_ids.shape[0]
    generated_len = generated_ids.shape[0]

    # Slice attention matrix: from generated tokens to input tokens
    generated_to_input_attn = avg_attn[-generated_len:, :input_len]

    # Sum attentions from all generated tokens to each input token
    aggregated_scores += generated_to_input_attn.sum(dim=0)
    passes += (generated_to_input_attn > 0).sum(dim=0)

    # Normalize contributions by dividing by the number of passes (to account for zero divisions)
    token_contributions = aggregated_scores / passes.clamp(min=1)

    print("\nToken Contributions to the Generated Response:")
    for token, contrib in zip(input_tokens_clean, token_contributions):
        print(f"Token: {token:15} | Contribution: {contrib.item():.4f}")

    # Filtering: Exclude tokens that match keywords and apply either z-score or quartile-based filtering
    token_contributions_np = token_contributions.numpy()

    # Apply keyword filtering
    keyword_mask = torch.tensor([token not in keywords for token in input_tokens_clean])

    if method == "z-score":
        # Z-score normalization
        mean = np.mean(token_contributions_np)
        std = np.std(token_contributions_np)
        z_scores = (token_contributions_np - mean) / (std if std > 0 else 1)
        print(f"\nZ-scores: {z_scores}")
        
        # Keep only tokens where |z| <= z_threshold
        z_mask = np.abs(z_scores) <= z_threshold
        filter_condition = keyword_mask & torch.tensor(z_mask)

    elif method == "quartile":
        # Quartile-based filtering
        Q1 = np.percentile(token_contributions_np, 25)
        Q3 = np.percentile(token_contributions_np, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        print(f"\nQuartiles: Q1={Q1}, Q3={Q3}, IQR={IQR}")
        print(f"Filtering with bounds: {lower_bound} <= contributions <= {upper_bound}")
        
        # Keep only tokens within the IQR bounds
        quartile_mask = (token_contributions_np >= lower_bound) & (token_contributions_np <= upper_bound)
        filter_condition = keyword_mask & torch.tensor(quartile_mask)

    else:
        raise ValueError(f"Invalid method '{method}'. Choose either 'z-score' or 'quartile'.")

    # Apply the filters and retain only the tokens and contributions that meet the condition
    token_contributions_filtered = token_contributions[filter_condition].numpy()
    filtered_tokens = [token for token, keep in zip(input_tokens_clean, filter_condition) if keep]

    # Check if filtering excluded all tokens
    if not filtered_tokens:
        print("\nAll tokens have been filtered out based on the provided keywords and outlier detection method.")
        return

    print("\nToken Contributions to the Generated Response (Filtered):")
    for token, contrib in zip(filtered_tokens, token_contributions_filtered):
        print(f"Token: {token:15} | Contribution: {contrib:.4f}")

    # Visualization: Bar chart of filtered token contributions
    plt.figure(figsize=(12, 6))
    contributions = token_contributions_filtered

    # Create a bar chart
    bars = plt.bar(filtered_tokens, contributions, color='skyblue')

    # Highlight the top 3 most attended tokens if more than 3 are present
    if len(contributions) >= 3:
        top_indices = contributions.argsort()[-3:][::-1]
        for idx in top_indices:
            bars[idx].set_color('orange')
    elif len(contributions) > 0:
        bars[0].set_color('orange')  # Highlight the first if fewer than 3 tokens

    plt.xlabel('Input Tokens')
    plt.ylabel('Average Contribution')
    plt.title(f'Input Token Contributions to Generated Response ({method.capitalize()} Filtered)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Add contribution values on top of the bars
    for bar, contrib in zip(bars, contributions):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{contrib:.2f}',
                 ha='center', va='bottom', fontsize=8)

    plt.show()

# 4. Main function to tie everything together
def main():
    if len(sys.argv) < 2:
        print("Usage: python run_llama_attention.py \"Your input text here\"")
        sys.exit(1)

    input_text = "User: Hello, My name is Kevin, can you tell me about the functions of attention? Llama: " #sys.argv[1]
    inputs, outputs = run_inference(input_text)
    analyze_and_visualize_attention(inputs, outputs)

if __name__ == "__main__":
    main()
