
#pip install huggingface-hub
#huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --include "original/*" --local-dir meta-llama/Llama-3.2-3B-Instruct
#pip install -U "huggingface_hub[cli]"
#huggingface-cli login
#python main.py
#pip install -U transformers --upgrade
#pip install accelerate
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Use pathlib to define the path to the model
model_path = Path("meta-llama") / "Llama-3.2-3B-Instruct"

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(str(model_path))  # Convert Path to string when passing to transformers
model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float16, device_map="auto")

model.config.output_attentions = True  # Enable attention tracking

conversation_history = ""  # To maintain conversation context

def generate_token_with_attention(input_ids):
    output = model(input_ids=input_ids, output_attentions=True)
    next_token_id = torch.argmax(output.logits[:, -1, :], dim=-1)  # Get next token
    attentions = output.attentions  # Attention across layers
    return next_token_id, attentions

def update_conversation(input_text):
    global conversation_history
    conversation_history += f"{input_text}\n"
    return tokenizer(conversation_history, return_tensors="pt").input_ids.to("cuda")

def generate_response(prompt, max_new_tokens=50):
    input_ids = update_conversation(prompt)
    generated_tokens, attention_scores = [], []

    for _ in range(max_new_tokens):
        next_token_id, attentions = generate_token_with_attention(input_ids)
        next_token = tokenizer.decode(next_token_id)
        
        generated_tokens.append(next_token)
        attention_scores.append(attentions)

        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
        if next_token_id == tokenizer.eos_token_id:
            break

    conversation_history += "".join(generated_tokens) + "\n"
    return "".join(generated_tokens), attention_scores

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        response, attentions = generate_response(user_input)
        print(f"Llama: {response}")
        print(f"Attention for last token: {attentions[-1]}")  # Inspect the attention for the last token


