import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.preprocess import load_shakespeare_data
from models.rag_query_optimizer import RAGQueryOptimizer
from configs.config import CONFIG

def train_model(model, tokenizer):
    # Load data
    sequences, stoi, itos = load_shakespeare_data(seq_len=CONFIG['seq_len'])
    train_loader = DataLoader(sequences, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Initialize RAG model with tokenizer vocabulary size
    rag_model = RAGQueryOptimizer(vocab_size=len(stoi), embedding_dim=CONFIG['embedding_dim'], top_n=5)
    optimizer = optim.Adam(rag_model.parameters(), lr=CONFIG['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()  # Define your loss function
    
    rag_model.train()
    for epoch in range(CONFIG['epochs']):
        for batch in train_loader:
            query_tokens, context_tokens, target_tokens = batch
            
            optimizer.zero_grad()
            
            # Forward pass through the RAG model
            reconstructed_query = rag_model(query_tokens, context_tokens)
            
            # Compute loss
            loss = criterion(reconstructed_query.view(-1, reconstructed_query.size(-1)), target_tokens.view(-1))
            loss.backward()
            
            optimizer.step()

        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}, Loss: {loss.item()}")

    print("Training complete.")
