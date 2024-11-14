import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from itertools import permutations

class DocumentReindexingExhaustive:
    def __init__(self, lm_model_name="meta-llama/Llama-2-7b-hf"):
        # Load the LLaMA 2 model and tokenizer
        self.lm_model = LlamaForCausalLM.from_pretrained(lm_model_name).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(lm_model_name)

    def reorder_chunks_exhaustive(self, query, chunks, prompt_text):
        """
        Exhaustively reorder chunks to maximize next-token probability.
        
        Args:
            query: Query tensor (batch_size, query_len, embed_dim)
            chunks: Document chunks tensor (batch_size, num_chunks, chunk_len, embed_dim)
            prompt_text: Text query to guide token prediction
        
        Returns:
            optimal_permutation: The permutation of chunks that maximizes the next-token probability.
            max_probability: The maximum probability achieved for the next token.
        """
        batch_size, num_chunks, chunk_len, embed_dim = chunks.size()
        
        # Flatten chunks for easier manipulation and tokenization
        chunk_texts = [
            self.tokenizer.decode(chunk.argmax(dim=-1).flatten().tolist())
            for chunk in chunks[0]  # Assuming batch_size = 1 for simplicity
        ]

        max_probability = -float('inf')
        optimal_permutation = None

        # Process prompt text to create a context
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(chunks.device)

        # Try every permutation of the chunks
        for perm in permutations(chunk_texts):
            # Concatenate query and current permutation of chunks
            permuted_text = prompt_text + " " + " ".join(perm)
            permuted_ids = self.tokenizer(permuted_text, return_tensors="pt").input_ids.to(chunks.device)
            
            # Pass through the model
            with torch.no_grad():
                outputs = self.lm_model(permuted_ids)
                next_token_logits = outputs.logits[:, -1, :]  # Get the logits for the next token

            # Compute probability of the most likely next token
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            predicted_token = next_token_probs.argmax(dim=-1)
            predicted_token_prob = next_token_probs[0, predicted_token].item()

            # Update optimal permutation if current one is better
            if predicted_token_prob > max_probability:
                max_probability = predicted_token_prob
                optimal_permutation = perm

        return optimal_permutation, max_probability


# Example Usage
batch_size = 1
num_chunks = 3  # Reduce for simplicity; exhaustive search scales factorially
chunk_len = 10
embed_dim = 256

# Generate dummy data
documents = torch.randn(batch_size, num_chunks, chunk_len, embed_dim)
query = torch.randn(batch_size, 10, embed_dim)
prompt_text = "The future of AI is"

# Initialize the model
model = DocumentReindexingExhaustive(lm_model_name="meta-llama/Llama-2-7b-hf")

# Get the optimal reordering
optimal_perm, max_prob = model.reorder_chunks_exhaustive(query, documents, prompt_text)
print("Optimal Permutation:", optimal_perm)
print("Max Probability:", max_prob)
