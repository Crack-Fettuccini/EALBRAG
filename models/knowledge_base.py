# models/knowledge_base.py
import os
import torch
from transformers import AutoTokenizer
from models.embedder import TokenEmbedder
import torch.nn.functional as F

class KnowledgeBase:
    def __init__(self, documents_path, embedder_model_name, device):
        """
        Initialize the Knowledge Base by loading and embedding documents.
        :param documents_path: Path to the folder containing document text files.
        :param embedder_model_name: Model name for the embedder.
        :param device: torch.device (cpu or cuda)
        """
        self.documents_path = documents_path
        self.device = device
        self.documents = self.load_documents()
        self.tokenizer = AutoTokenizer.from_pretrained(embedder_model_name, use_fast=True)
        self.embedder = TokenEmbedder(vocab_size=self.tokenizer.vocab_size, embedding_dim=128)  # Ensure embedding_dim matches CONFIG
        self.embedder.to(device)
        self.embedder.eval()
        self.embeddings = self.embed_documents()

    def load_documents(self):
        """
        Load all text documents from the documents directory.
        :return: List of document strings.
        """
        documents = []
        for filename in os.listdir(self.documents_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.documents_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
        print(f"Loaded {len(documents)} documents from {self.documents_path}.")
        return documents

    def embed_documents(self):
        """
        Embed all loaded documents.
        :return: Tensor of shape (num_docs, embed_dim)
        """
        embeddings = []
        with torch.no_grad():
            for doc in self.documents:
                tokens = self.tokenizer.encode(doc, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                embedded = self.embedder(tokens)  # (1, seq_len, embed_dim)
                doc_embedding = embedded.mean(dim=1)  # (1, embed_dim)
                doc_embedding = F.normalize(doc_embedding, p=2, dim=1)
                embeddings.append(doc_embedding.squeeze(0))
        embeddings = torch.stack(embeddings)  # (num_docs, embed_dim)
        print(f"Embedded {embeddings.size(0)} documents.")
        return embeddings  # (num_docs, embed_dim)

    def get_embeddings(self):
        """
        Get the precomputed embeddings.
        :return: Tensor of shape (num_docs, embed_dim)
        """
        return self.embeddings
