import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from data.preprocess import load_shakespeare_data, ShakespeareDataset
from models.rag_query_optimizer import RAGQueryOptimizer
from models.HyDE import HyDE
from models.extrapolator import Extrapolator
from models.memory import Memory
from models.sliding_window import SlidingWindowManager
from configs.config import CONFIG

def train_model(generator_model, tokenizer, device, knowledge_base, memory, sliding_window):
    """
    Train the RAG model with Sliding Window and Memory mechanisms.
    :param generator_model: The Llama language model.
    :param tokenizer: The tokenizer.
    :param device: torch.device
    :param knowledge_base: KnowledgeBase instance.
    :param memory: Memory instance.
    :param sliding_window: SlidingWindowManager instance.
    """
    # Load data
    print("Loading training data...")
    sequences, stoi, itos = load_shakespeare_data(seq_len=CONFIG['seq_len'])
    dataset = ShakespeareDataset(sequences)
    train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Initialize RAG Query Optimizer
    print("Initializing RAG Query Optimizer...")
    rag_model = RAGQueryOptimizer(
        vocab_size=len(stoi),
        embedding_dim=CONFIG['embedding_dim'],
        knowledge_base_embeddings=knowledge_base.get_embeddings(),
        device=device,
        top_n=CONFIG['top_n'],
        top_k=CONFIG['top_k'],
        window_size=CONFIG['window_size'],
        memory_size=CONFIG['memory_size']
    )
    rag_model.to(device)

    optimizer = optim.Adam(rag_model.parameters(), lr=CONFIG['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    print("Starting training...")
    rag_model.train()
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            query_tokens, target_tokens = batch
            query_tokens = query_tokens.to(device)  # (batch, seq_len)
            target_tokens = target_tokens.to(device)  # (batch, seq_len)

            optimizer.zero_grad()

            # Forward pass through RAG optimizer
            reconstructed_query, rag_scores, retrieved_docs = rag_model(query_tokens, query_tokens, tokenizer)

            # Placeholder: In practice, define a suitable loss function that aligns with your task
            # Here, we assume reconstructed_query should predict target_tokens
            # This requires reconstructed_query to output logits over the vocabulary
            # Modify RAGQueryOptimizer accordingly if necessary

            # For demonstration, we'll use CrossEntropyLoss between reconstructed_query and target_tokens
            # Ensure reconstructed_query has shape (batch * window_size, vocab_size)
            # and target_tokens have shape (batch * window_size)
            # This may require adjusting the output of RAGQueryOptimizer

            # Assuming reconstructed_query is token IDs, which is not suitable for CrossEntropyLoss
            # Instead, you need to have the model output logits
            # Here, we use a dummy loss by treating reconstructed_query as target for demonstration

            # Dummy Loss (Replace with appropriate loss)
            loss = criterion(reconstructed_query.view(-1, reconstructed_query.size(-1)), target_tokens.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % CONFIG['log_interval'] == 0:
                avg_loss = total_loss / CONFIG['log_interval']
                print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                total_loss = 0

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(CONFIG['checkpoints_dir'], f"rag_epoch_{epoch+1}.pth")
        torch.save(rag_model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # Save the final model
    final_model_path = CONFIG['rag_model_path']
    torch.save(rag_model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved at {final_model_path}")
