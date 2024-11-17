import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import math
import logging
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class HyDE:
    def __init__(self, 
                 model: LlamaForCausalLM, 
                 tokenizer: AutoTokenizer, 
                 device: torch.device, 
                 max_length: int = 200, 
                 num_documents: int = 1, 
                 temperature: float = 0.7, 
                 top_p: float = 0.9, 
                 top_k: int = 50, 
                 repetition_penalty: float = 1.0, 
                 alpha: float = 0.7):
        """
        Initialize the HyDE (Hypothetical Document Embeddings) module.
        
        Args:
            model (LlamaForCausalLM): Pretrained language model.
            tokenizer (AutoTokenizer): Tokenizer for the model.
            device (torch.device): Device for computation.
            max_length (int): Maximum length of generated documents.
            num_documents (int): Number of hypothetical documents to generate.
            temperature (float): Sampling temperature for diversity.
            top_p (float): Nucleus sampling probability.
            top_k (int): Top-k sampling parameter.
            repetition_penalty (float): Penalty for repeated tokens.
            alpha (float): Weight for combining relevance and readability scores.
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.num_documents = num_documents
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.alpha = alpha

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate the perplexity of a given text using the model.
        
        Args:
            text (str): Input text for perplexity calculation.
        
        Returns:
            float: Perplexity score (lower is better).
        """
        try:
            encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings.input_ids)
                loss = outputs.loss
            return math.exp(loss.item())
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')

    def rank_documents(self, documents: List[str], query_embedding: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Rank documents based on combined relevance and readability scores.
        
        Args:
            documents (List[str]): List of generated documents.
            query_embedding (torch.Tensor): Embedding of the query for relevance scoring.
        
        Returns:
            List[Tuple[str, float]]: Ranked documents with their scores.
        """
        rankings = []
        for doc in documents:
            try:
                doc_embedding = self.tokenizer(doc, return_tensors="pt").to(self.device)
                relevance_score = torch.cosine_similarity(query_embedding, doc_embedding, dim=-1).mean().item()
                readability_score = 1 / self.calculate_perplexity(doc)
                combined_score = self.alpha * relevance_score + (1 - self.alpha) * readability_score
                rankings.append((doc, combined_score))
            except Exception as e:
                logger.warning(f"Error ranking document: {e}")
                continue

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def generate_hypothetical_documents(self, query: str, additional_context: Optional[str] = None) -> List[str]:
        """
        Generate hypothetical documents relevant to the query and rank them.
        
        Args:
            query (str): Input query.
            additional_context (Optional[str]): Additional context for generation.
        
        Returns:
            List[str]: Ranked list of generated documents.
        """
        try:
            prompt = f"Generate a detailed document relevant to the following query: {query}\n\nDocument:"
            if additional_context:
                prompt = f"{additional_context}\n\n{prompt}"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=self.max_length,
                num_return_sequences=self.num_documents,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True).split("Document:")[1].strip() 
                for output in outputs
            ]

            # Filter out empty or incomplete documents
            generated_texts = [text for text in generated_texts if len(text) > 0]

            query_embedding = self.tokenizer(query, return_tensors="pt").to(self.device)

            # Rank and return
            ranked_docs = self.rank_documents(generated_texts, query_embedding)
            return [doc for doc, _ in ranked_docs]
        
        except Exception as e:
            logger.error(f"Error generating documents: {e}")
            return []

    def generate_and_return_token_ids(self, query: str) -> List[Tuple[str, torch.Tensor]]:
        """
        Generate hypothetical documents and return ranked texts with their token IDs.
        
        Args:
            query (str): Input query.
        
        Returns:
            List[Tuple[str, torch.Tensor]]: Ranked documents and their tokenized IDs.
        """
        try:
            ranked_texts = self.generate_hypothetical_documents(query)
            tokenized_texts = [(text, self.tokenizer.encode(text, return_tensors="pt").to(self.device)) for text in ranked_texts]
            return tokenized_texts
        except Exception as e:
            logger.error(f"Error generating tokenized outputs: {e}")
            return []
