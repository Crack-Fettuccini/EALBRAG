import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load the existing model weights from the .pth file
consolidated_model_path = "meta-llama\\Meta-Llama-3-8B-Instruct\\original\\consolidated.00.pth"  # Update this with your actual path
model_name = "meta-llama\\Meta-Llama-3-8B-Instruct\\original"  # Use the correct model name

# Initialize the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Initialize the model architecture
model = LlamaForCausalLM.from_pretrained(model_name)

# Load the model parameters from the .pth file
checkpoint = torch.load(consolidated_model_path, map_location='cpu')
model.load_state_dict(checkpoint)

# Save the model in Hugging Face format (as .bin)
output_directory = "meta-llama\\Meta-Llama-3-8B-Instruct\\original"  # Desired output directory
model.save_pretrained(output_directory)
tokenizer.save_pretrained(output_directory)

print(f"Model has been saved to {output_directory} in Hugging Face format.")
