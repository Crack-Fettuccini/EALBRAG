import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from training.train import train_model
from training.generate import generate_text
from configs.config import CONFIG
from models.knowledge_base import KnowledgeBase
from models.memory import Memory
from models.sliding_window import SlidingWindowManager

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Shakespeare RAG with HyDE, Extrapolator, Sliding Window, and Memory')
    parser.add_argument('--task', choices=['train', 'generate'], required=True, help='Task to perform: train or generate')
    parser.add_argument('--start_text', type=str, help='Text to start generation')
    parser.add_argument('--conversation_history', type=str, nargs='*', help='Past conversation history as separate strings')
    args = parser.parse_args()

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
    model.to(device)
    model.eval()

    # Initialize Knowledge Base
    print("Initializing Knowledge Base...")
    knowledge_base = KnowledgeBase(documents_path="data/documents", embedder_model_name=model_name, device=device)

    # Initialize Memory
    print("Initializing Memory...")
    memory = Memory(embed_dim=CONFIG['embedding_dim'], memory_size=CONFIG['memory_size'], device=device)

    # Initialize Sliding Window Manager
    print("Initializing Sliding Window Manager...")
    sliding_window = SlidingWindowManager(window_size=CONFIG['window_size'], device=device)

    if args.task == 'train':
        print("Starting Training Process...")
        train_model(model, tokenizer, device, knowledge_base, memory, sliding_window)
    elif args.task == 'generate':
        if args.start_text:
            conversation_history = args.conversation_history if args.conversation_history else []
            print("Starting Text Generation Process...")
            generated_text = generate_text(
                generator_model=model,
                tokenizer=tokenizer,
                prompt=args.start_text,
                device=device,
                conversation_history=conversation_history,
                knowledge_base=knowledge_base,
                memory=memory,
                sliding_window=sliding_window
            )
            print(f"\nGenerated Text:\n{generated_text}\n")
        else:
            print("Error: Please provide --start_text for text generation.")
            sys.exit(1)

if __name__ == '__main__':
    main()
