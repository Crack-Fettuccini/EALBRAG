import os
import torch
from transformers import AutoTokenizer, AutoModel
from models.embedder import TokenEmbedder
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics.pairwise import cosine_similarity
import logging

class KnowledgeBase:
    def __init__(self, documents_path, embedder_model_name, device, embedding_dim=128, batch_size=8, cache_file=None, max_length=512):
        """
        Initialize the Knowledge Base by loading and embedding documents.
        :param documents_path: Path to the folder containing document text files.
        :param embedder_model_name: Model name for the embedder or a transformer-based model.
        :param device: torch.device (cpu or cuda)
        :param embedding_dim: Embedding dimension if using custom TokenEmbedder.
        :param batch_size: Number of documents to embed at once for batch processing.
        :param cache_file: Path to save/load precomputed embeddings for faster initialization.
        :param max_length: Maximum token length for document embeddings (handles truncation).
        """
        self.documents_path = documents_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_file = cache_file
        self.tokenizer = AutoTokenizer.from_pretrained(embedder_model_name, use_fast=True)
        
        # Check if we're using a transformer-based model or custom embedder
        if "transformer" in embedder_model_name.lower():
            self.embedder = AutoModel.from_pretrained(embedder_model_name).to(self.device)
        else:
            self.embedder = TokenEmbedder(vocab_size=self.tokenizer.vocab_size, embedding_dim=embedding_dim).to(self.device)

        self.embedder.eval()  # Set embedder to evaluation mode

        # Attempt to load cached embeddings, otherwise, load documents and compute embeddings
        if cache_file and os.path.exists(cache_file):
            self.load_embeddings(cache_file)
        else:
            self.documents, self.document_metadata = self.load_documents()
            self.embeddings = self.embed_documents()

    def load_documents(self):
        """
        Load all text documents from the documents directory and gather metadata.
        :return: Tuple of (document strings list, metadata list).
        """
        documents = []
        metadata = []
        for filename in os.listdir(self.documents_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.documents_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            documents.append(content)
                            metadata.append({"filename": filename, "filepath": filepath})
                except Exception as e:
                    logging.error(f"Error reading {filepath}: {e}")
        logging.info(f"Loaded {len(documents)} documents from {self.documents_path}.")
        return documents, metadata

    def embed_documents(self):
        """
        Embed all loaded documents in batches. Handle chunking for long documents.
        :return: Tensor of shape (num_docs, embed_dim)
        """
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(self.documents), self.batch_size):
                batch_docs = self.documents[i:i + self.batch_size]
                
                # Tokenize documents and handle chunking for long documents
                batch_tokens = [self.tokenize_and_chunk(doc) for doc in batch_docs]
                batch_tokens_flat = [token for tokens in batch_tokens for token in tokens]  # Flatten list of chunks
                
                # Pad sequences to ensure they have the same shape
                batch_tokens_padded = pad_sequence(batch_tokens_flat, batch_first=True)
                batch_tokens_padded = batch_tokens_padded.to(self.device)
                
                # Embed the batch
                if isinstance(self.embedder, TokenEmbedder):
                    embedded = self.embedder(batch_tokens_padded)  # Custom TokenEmbedder
                else:
                    embedded = self.embedder(batch_tokens_padded).last_hidden_state  # Transformer-based model

                # Average over the sequence length (mean pooling) for document-level embedding
                doc_embeddings = embedded.mean(dim=1)
                doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)  # Normalize embeddings
                
                # Re-assemble document embeddings from chunk embeddings
                batch_embeddings = []
                for tokens in batch_tokens:
                    # Average over the chunk embeddings to get the full document embedding
                    chunk_embeds = doc_embeddings[:len(tokens)].mean(dim=0)
                    batch_embeddings.append(chunk_embeds)
                    doc_embeddings = doc_embeddings[len(tokens):]  # Trim the processed chunks

                embeddings.extend(batch_embeddings)

        embeddings = torch.stack(embeddings)  # Stack embeddings into a tensor
        logging.info(f"Embedded {len(embeddings)} documents.")
        return embeddings

    def tokenize_and_chunk(self, document):
        """
        Tokenize and chunk a document if it exceeds the max token length.
        :param document: A document string.
        :return: List of tokenized chunks (each chunk is tensor).
        """
        tokens = self.tokenizer.encode(document, return_tensors="pt", truncation=False)[0]  # Get raw tokens
        chunks = torch.split(tokens, self.max_length)  # Split into chunks of max_length
        return [chunk.unsqueeze(0) for chunk in chunks]  # Add batch dimension

    def get_embeddings(self):
        """
        Get the precomputed embeddings.
        :return: Tensor of shape (num_docs, embed_dim)
        """
        return self.embeddings

    def get_document_metadata(self):
        """
        Get the metadata of the documents (filenames, paths, etc.).
        :return: List of metadata dictionaries.
        """
        return self.document_metadata

    def save_embeddings(self, file_path):
        """
        Save the document embeddings to a file for later use.
        :param file_path: Path to the output file.
        """
        torch.save(self.embeddings, file_path)
        logging.info(f"Saved embeddings to {file_path}.")

    def load_embeddings(self, file_path):
        """
        Load precomputed document embeddings from a file.
        :param file_path: Path to the embeddings file.
        """
        self.embeddings = torch.load(file_path)
        logging.info(f"Loaded embeddings from {file_path}.")

    def find_similar_documents(self, query, top_k=5):
        """
        Find the most similar documents to a given query using cosine similarity.
        :param query: Query string to search for.
        :param top_k: Number of top similar documents to return.
        :return: List of tuples (document metadata, similarity score)
        """
        # Tokenize and embed the query
        with torch.no_grad():
            query_tokens = self.tokenize_and_chunk(query)
            query_tokens_padded = pad_sequence(query_tokens, batch_first=True).to(self.device)
            
            # Embed the query using the same model
            if isinstance(self.embedder, TokenEmbedder):
                query_embedding = self.embedder(query_tokens_padded).mean(dim=1)
            else:
                query_embedding = self.embedder(query_tokens_padded).last_hidden_state.mean(dim=1)
                
            query_embedding = F.normalize(query_embedding.mean(dim=0), p=2, dim=0).unsqueeze(0)  # Average over chunks, then normalize

        # Calculate cosine similarities between query and all document embeddings
        cosine_sim = cosine_similarity(query_embedding.cpu().numpy(), self.embeddings.cpu().numpy())[0]
        
        # Get the top K most similar documents
        top_k_indices = cosine_sim.argsort()[-top_k:][::-1]  # Indices of top K documents
        top_k_similarities = cosine_sim[top_k_indices]

        # Retrieve metadata for the top K documents
        similar_documents = [(self.document_metadata[idx], top_k_similarities[i]) for i, idx in enumerate(top_k_indices)]
        return similar_documents

    def display_similar_documents(self, query, top_k=5):
        """
        Display the top-k similar documents along with their similarity scores.
        :param query: Query string.
        :param top_k: Number of top similar documents to display.
        """
        similar_documents = self.find_similar_documents(query, top_k)
        for i, (doc_meta, score) in enumerate(similar_documents):
            print(f"Rank {i+1}: {doc_meta['filename']} (Similarity: {score:.4f})")
