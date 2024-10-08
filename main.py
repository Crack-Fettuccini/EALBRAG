import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from training.train import train_model
from training.generate import generate_text
from configs.config import CONFIG

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
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
model.eval()

def main():
    # Example of a simple CLI interface
    while True:
        print("Choose an option:")
        print("1: Train the model")
        print("2: Generate text")
        print("3: Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            train_model(model, tokenizer)
        elif choice == '2':
            prompt = input("Enter your prompt: ")
            generated_text = generate_text(model, tokenizer, prompt)
            print(f"Generated Text: {generated_text}")
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
