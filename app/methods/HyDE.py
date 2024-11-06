import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import math

class HyDE:
    def __init__(self, model, tokenizer, device, max_length=200, num_documents=1, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.0, alpha=0.7):
        """
        Initialize the HyDE module with a language model and tokenizer, adding LLM readability scoring.
        :param model: Pretrained language model for generation
        :param tokenizer: Tokenizer corresponding to the model
        :param device: torch.device (cpu or cuda)
        :param max_length: Maximum length of generated hypothetical documents
        :param num_documents: Number of hypothetical documents to generate
        :param temperature: Sampling temperature for generation diversity
        :param top_p: Nucleus sampling parameter
        :param top_k: Top-k sampling parameter
        :param repetition_penalty: Penalty for repeated phrases to enhance diversity
        :param alpha: Weighting factor for relevance vs readability in ranking
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.num_documents = num_documents
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.alpha = alpha  # Controls relevance vs. readability weight in scoring

    def calculate_perplexity(self, text):
        """
        Calculate perplexity of a document to measure LLM readability.
        :param text: Document text
        :return: Perplexity score (lower is more readable)
        """
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings.input_ids)
            loss = outputs.loss
        return math.exp(loss.item())

    def rank_documents(self, documents, query_embedding):
        """
        Rank documents by combining relevance and readability scores.
        :param documents: List of generated document texts
        :param query_embedding: Embedding vector of the query for relevance scoring
        :return: Ranked list of (document, score) based on combined readability and relevance
        """
        rankings = []
        for doc in documents:
            # Calculate relevance as cosine similarity to the query
            doc_embedding = self.tokenizer(doc, return_tensors="pt").to(self.device)
            relevance_score = torch.cosine_similarity(query_embedding, doc_embedding, dim=1).item()
            
            # Calculate readability as the inverse of perplexity
            readability_score = 1 / self.calculate_perplexity(doc)

            # Weighted combination of relevance and readability
            combined_score = self.alpha * relevance_score + (1 - self.alpha) * readability_score
            rankings.append((doc, combined_score))

        # Sort documents by combined score in descending order
        rankings.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in rankings]

    def generate_hypothetical_documents(self, query, additional_context=None):
        """
        Generate hypothetical documents based on the query and rank by readability and relevance.
        :param query: Input query string
        :param additional_context: Optional additional context for generation
        :return: Ranked list of generated document strings
        """
        # Prepare the prompt
        prompt = f"Generate a detailed document relevant to the following query: {query}\n\nDocument:"
        if additional_context:
            prompt = f"{additional_context}\n\n{prompt}"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate documents
        outputs = self.model.generate(
            inputs,
            max_length=self.max_length,
            num_return_sequences=self.num_documents,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode the generated documents
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True).split("Document:")[1].strip() 
            for output in outputs
        ]

        # Remove any empty or incomplete documents
        generated_texts = [text for text in generated_texts if len(text) > 0]
        
        # Embed query for relevance comparison
        query_embedding = self.tokenizer(query, return_tensors="pt").to(self.device)
        
        # Rank documents by combined relevance and readability
        ranked_documents = self.rank_documents(generated_texts, query_embedding)
        
        return ranked_documents

    def generate_and_return_token_ids(self, query):
        """
        Generate hypothetical documents and return ranked text with token IDs for each document.
        :param query: Input query string
        :return: List of tuples (ranked_text, token_ids) with highest scoring documents
        """
        ranked_texts = self.generate_hypothetical_documents(query)
        tokenized_texts = [(text, self.tokenizer.encode(text, return_tensors="pt").to(self.device)) for text in ranked_texts]
        return tokenized_texts
