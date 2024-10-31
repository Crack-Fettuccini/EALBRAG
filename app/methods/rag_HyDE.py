import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from HyDE import HyDE
from rag_query_optimizer import RAGQueryOptimizer

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize HyDE model with pre-trained model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to(device)

# Instantiate HyDE with the model, tokenizer, and necessary generation parameters
hyde = HyDE(
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_length=200,
    num_documents=1,  # Number of hypothetical documents to generate
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.0
)

# Define the RAG Query Optimizer parameters
vocab_size = tokenizer.vocab_size
embedding_dim = 768  # Assumed dimension, set according to your specific model's config
knowledge_base_embeddings = torch.randn(1000, embedding_dim).to(device)  # Placeholder knowledge base embeddings

# Instantiate RAGQueryOptimizer with pre-initialized submodules and hyperparameters
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

# Function to perform RAG with HyDE-generated documents
def perform_rag_with_hyde(query):
    # Step 1: Generate hypothetical documents with HyDE based on the input query
    hypothetical_documents = hyde.generate_hypothetical_documents(query)

    # Step 2: Tokenize each document for processing in the RAG pipeline
    tokenized_docs = [tokenizer.encode(doc, return_tensors="pt").to(device) for doc in hypothetical_documents]

    # Step 3: Process each tokenized document with RAGQueryOptimizer
    rag_outputs = []
    for doc_tokens in tokenized_docs:
        # Prepare prompt tokens (if required, otherwise pass an empty tensor)
        prompt_tokens = torch.tensor([]).to(device)
        
        # Pass through RAG query optimizer
        reconstructed_query, rag_scores, retrieved_indices = rag_query_optimizer(
            query_tokens=doc_tokens, prompt_tokens=prompt_tokens, tokenizer=tokenizer
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
query = "Explain the impact of quantum computing on cryptography."
rag_results = perform_rag_with_hyde(query)

# Print the results
for idx, result in enumerate(rag_results):
    print(f"Reconstructed Query {idx+1}: {result['reconstructed_query']}")
    print(f"RAG Scores: {result['rag_scores']}")
    print(f"Retrieved Document Indices: {result['retrieved_indices']}")
    print("\n" + "="*50 + "\n")
