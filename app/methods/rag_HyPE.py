import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from HyPE import HyPE
from rag_query_optimizer import RAGQueryOptimizer

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize HyPE model with pre-trained model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to(device)

# Instantiate HyPE with the model, tokenizer, and necessary generation parameters
hype = HyPE(
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_length=200,
    num_ground_truths=1,  # Number of hypothetical profiles to generate
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.0
)

# Define the RAG Query Optimizer parameters
vocab_size = tokenizer.vocab_size
embedding_dim = 768  # Set according to your model's specific configuration
knowledge_base_embeddings = torch.randn(1000, embedding_dim).to(device)  # Placeholder embeddings for knowledge base

# Instantiate RAGQueryOptimizer with pre-initialized parameters and hyperparameters
rag_query_optimizer = RAGQueryOptimizer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    knowledge_base_embeddings=knowledge_base_embeddings,
    device=device,
    num_heads=8,
    top_n=5,
    top_k=5,
    window_size=512,
    memory_size=1000
)

# Function to perform RAG with HyPE-generated user profiles
def perform_rag_with_hype(conversation_history):
    """
    Perform the RAG process with HyPE-generated user profiles based on conversation history.
    :param conversation_history: List of conversation strings.
    :return: List of RAG-optimized results for generated user profiles.
    """
    # Step 1: Generate hypothetical user profiles with HyPE based on the conversation history
    hypothetical_profiles = hype.generate_hypothetical_ground_truths(conversation_history, retrieved_docs=[])

    # Step 2: Tokenize each profile for processing in the RAG pipeline
    tokenized_profiles = [tokenizer.encode(profile, return_tensors="pt").to(device) for profile in hypothetical_profiles]

    # Step 3: Process each tokenized profile with RAGQueryOptimizer
    rag_outputs = []
    for profile_tokens in tokenized_profiles:
        # Prepare prompt tokens (if required, otherwise pass an empty tensor)
        prompt_tokens = torch.tensor([]).to(device)
        
        # Pass through RAG query optimizer
        reconstructed_query, rag_scores, retrieved_indices = rag_query_optimizer(
            query_tokens=profile_tokens, prompt_tokens=prompt_tokens, tokenizer=tokenizer
        )
        
        # Decode and store the reconstructed query results
        reconstructed_text = tokenizer.decode(reconstructed_query[0], skip_special_tokens=True)
        rag_outputs.append({
            "reconstructed_query": reconstructed_text,
            "rag_scores": rag_scores,
            "retrieved_indices": retrieved_indices
        })

    return rag_outputs

# Example usage
conversation_history = [
    "I am very interested in the impact of blockchain on finance.",
    "I would like to learn about the role of artificial intelligence in healthcare.",
    "Do you have resources on data privacy and security trends?"
]

# Perform RAG with HyPE
rag_results = perform_rag_with_hype(conversation_history)

# Print the results
for idx, result in enumerate(rag_results):
    print(f"Reconstructed Query {idx+1}: {result['reconstructed_query']}")
    print(f"RAG Scores: {result['rag_scores']}")
    print(f"Retrieved Document Indices: {result['retrieved_indices']}")
    print("\n" + "="*50 + "\n")
