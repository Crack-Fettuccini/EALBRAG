# training/generate.py
import torch
from models.rag_query_optimizer import RAGQueryOptimizer
from models.HyDE import HyDE
from models.extrapolator import Extrapolator
from models.memory import Memory
from models.sliding_window import SlidingWindowManager
from configs.config import CONFIG

def generate_text(generator_model, tokenizer, prompt, device, conversation_history, knowledge_base, memory, sliding_window):
    """
    Generate text using the RAG optimizer, HyDE, and Query Extrapolator.
    :param generator_model: The Llama language model.
    :param tokenizer: The tokenizer.
    :param prompt: The input prompt string.
    :param device: torch.device
    :param conversation_history: List of past conversation strings.
    :param knowledge_base: KnowledgeBase instance.
    :param memory: Memory instance.
    :param sliding_window: SlidingWindowManager instance.
    :return: Generated text string.
    """
    from transformers import AutoModelForCausalLM

    # Initialize Extrapolator
    print("Initializing Extrapolator...")
    extrapolator = Extrapolator(generator_model, tokenizer, device, max_length=CONFIG['max_length_extrapolator'])

    # Generate extrapolated query based on conversation history and current prompt
    print("Generating extrapolated query...")
    extrapolated_query = extrapolator.generate_extrapolated_query(conversation_history, prompt)
    print(f"Extrapolated Query: {extrapolated_query}")

    # Initialize HyDE
    print("Generating hypothetical documents with HyDE...")
    hyde = HyDE(generator_model, tokenizer, device, max_length=CONFIG['max_length_hyde'], num_documents=CONFIG['num_hyde_docs'])
    hypothetical_docs = hyde.generate_hypothetical_documents(extrapolated_query)
    print(f"Generated Hypothetical Documents: {hypothetical_docs}")

    # Retrieve actual documents based on the extrapolated query
    print("Retrieving documents from Knowledge Base...")
    # Encode the extrapolated query
    query_tokens = tokenizer.encode(extrapolated_query, return_tensors="pt").to(device)
    # Assume RAGQueryOptimizer handles retrieval
    # Initialize RAG Query Optimizer (Load trained model if necessary)
    rag_model = RAGQueryOptimizer(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=CONFIG['embedding_dim'],
        knowledge_base_embeddings=knowledge_base.get_embeddings(),
        device=device,
        top_n=CONFIG['top_n'],
        top_k=CONFIG['top_k'],
        window_size=CONFIG['window_size'],
        memory_size=CONFIG['memory_size']
    )
    rag_model.to(device)
    rag_model.eval()

    # Forward pass to get reconstructed query and retrieved documents
    with torch.no_grad():
        reconstructed_query, rag_scores, retrieved_docs_indices = rag_model(query_tokens, query_tokens, tokenizer)

    # Retrieve documents based on indices
    retrieved_documents = [knowledge_base.documents[idx] for idx in retrieved_docs_indices[0].cpu().tolist()]
    print(f"Retrieved Documents: {retrieved_documents}")

    # Combine hypothetical documents and retrieved documents
    combined_documents = hypothetical_docs + retrieved_documents

    # Prepare the final prompt by combining the original prompt with retrieved information
    retrieved_text = "\n".join(combined_documents)
    final_prompt = f"{prompt}\n\nRelevant Information:\n{retrieved_text}\n\nAnswer:"

    # Tokenize the final prompt
    input_ids = tokenizer.encode(final_prompt, return_tensors="pt").to(device)

    # Generate text using the generator model
    print("Generating response with the generator model...")
    with torch.no_grad():
        output_ids = generator_model.generate(
            input_ids,
            max_length=CONFIG['max_length_generate'],
            num_return_sequences=1,
            temperature=CONFIG['temperature'],
            top_p=CONFIG['top_p'],
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text
