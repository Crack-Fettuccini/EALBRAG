#pip install huggingface-hub
#huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --include "original/*" --local-dir meta-llama/Llama-3.2-3B-Instruct
#pip install -U "huggingface_hub[cli]"
#huggingface-cli login
#pip install -U transformers --upgrade
#pip install accelerate
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Use pathlib to define the path to the model
model_path = "NousResearch/Llama-3.2-1B"

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(str(model_path))
model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
).to(device)

model.eval()  # Set model to evaluation mode

conversation_history = ""  # To maintain conversation context

def generate_token(input_ids):
    with torch.no_grad():  # Disable gradient tracking for faster inference
        output = model(input_ids=input_ids)
        next_token_id = torch.argmax(output.logits[:, -1, :], dim=-1)  # Get next token
    return next_token_id

def update_conversation(input_text):
    global conversation_history
    conversation_history += f"{input_text}\n"
    return tokenizer(conversation_history, return_tensors="pt").input_ids.to(device)

def generate_response(prompt, max_new_tokens=50):
    input_ids = update_conversation(prompt)
    generated_tokens = []

    for _ in range(max_new_tokens):
        next_token_id = generate_token(input_ids)
        next_token = tokenizer.decode(next_token_id)

        generated_tokens.append(next_token)

        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
        if next_token_id == tokenizer.eos_token_id:
            break

    update_conversation("".join(generated_tokens) + "\n")
    return "".join(generated_tokens)

if __name__ == "__main__":
    while True:
        user_input = input("You:")  # For testing, you can replace this with input() for user input
        response = generate_response(user_input+"Llama:")
        print(f"{response}")
