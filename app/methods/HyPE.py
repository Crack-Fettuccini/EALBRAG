import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import yaml
import os
import logging
from typing import List
from utils.cache import SimpleCache
from utils.privacy import PrivacySanitizer
from database.user_profile_db import UserProfileDB
from utils.summarizer import Summarizer  # Summarization module
from utils.trend_analyzer import TrendAnalyzer  # Trend analysis module
from peft import LoRA  # Parameter-efficient tuning (PEFT)
from sklearn.metrics.pairwise import cosine_similarity

class HyPE:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the HyPE module with configurations, model, tokenizer, cache, and database.
        :param config_path: Path to the configuration YAML file.
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self.device = torch.device(self.config['model']['device'] if torch.cuda.is_available() else "cpu")
        self.model = LlamaForCausalLM.from_pretrained(self.config['model']['name']).to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config['model']['name'])
        self.max_length = self.config['model']['max_length']
        self.num_ground_truths = self.config['model']['num_ground_truths']
        self.temperature = self.config['model']['temperature']
        self.top_p = self.config['model']['top_p']
        
        # Initialize PEFT (LoRA)
        if self.config['peft']['enabled']:
            self.model = LoRA(self.model, r=self.config['peft']['rank'], alpha=self.config['peft']['alpha'])
            self.logger.info("PEFT (LoRA) initialized.")
        
        # Initialize PrivacySanitizer
        if self.config['privacy']['enabled']:
            self.privacy_sanitizer = PrivacySanitizer(self.config['privacy'])
            self.logger.info("PrivacySanitizer initialized.")
        else:
            self.privacy_sanitizer = None
            self.logger.info("PrivacySanitizer is disabled.")
        
        # Initialize cache
        if self.config['cache']['enabled']:
            self.cache = SimpleCache(cache_dir=self.config['cache']['cache_dir'], max_size=self.config['cache']['max_cache_size'])
            self.logger.info("Caching is enabled.")
        else:
            self.cache = None
            self.logger.info("Caching is disabled.")
        
        # Initialize database
        self.db = UserProfileDB(self.config['database']['connection_string'])
        self.logger.info("UserProfileDB initialized.")
        
        # Initialize summarizer and trend analyzer
        self.summarizer = Summarizer(self.config['summarization'])
        self.trend_analyzer = TrendAnalyzer(self.config['trend_analysis'])
        
        self.logger.info("HyPE initialized successfully.")
    
    def setup_logging(self):
        """
        Setup logging based on configuration.
        """
        log_level_str = self.config.get('logging', {}).get('level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_file = self.config.get('logging', {}).get('log_file', 'hype.log')
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def generate_hypothetical_ground_truths(self, conversation_history: List[str], retrieved_docs: List[str]) -> List[str]:
        """
        Generate Hypothetical Ground Truths based on conversation history and retrieved documents.
        Recheck existing data and only update if the confidence score is high enough.
        """
        # Sanitize inputs for privacy
        if self.privacy_sanitizer:
            sanitized_history = [self.privacy_sanitizer.sanitize_input(text) for text in conversation_history]
            sanitized_docs = [self.privacy_sanitizer.sanitize_input(doc) for doc in retrieved_docs]
            self.logger.debug("Input texts have been sanitized.")
        else:
            sanitized_history = conversation_history
            sanitized_docs = retrieved_docs
        
        # Summarize conversation history
        summarized_history = self.summarizer.summarize_conversation(sanitized_history)
        self.logger.debug("Conversation history summarized.")
        
        # Perform trend analysis
        trends = self.trend_analyzer.analyze_trends(sanitized_history)
        self.logger.debug("Trend analysis complete.")
        
        # Combine summarized history, trends, and retrieved documents
        context = "\n".join(summarized_history) + "\n" + "\n".join(sanitized_docs) + "\n" + "\n".join(trends)
        
        # Check cache
        cache_key = self._generate_cache_key(context)
        if self.cache and self.cache.contains(cache_key):
            self.logger.info("Cache hit for the given context.")
            return self.cache.get(cache_key)
        
        # Prepare the prompt for generating Hypothetical Ground Truths
        prompt = (
            f"You are an intelligent assistant analyzing the following summarized conversation and retrieved documents.\n\n"
            f"Conversation History:\n{context}\n\n"
            f"Based on the above, generate a detailed Hypothetical Ground Truth that summarizes the user's profile, interests, and relevant personal information to enhance future responses.\n\n"
            f"Hypothetical Ground Truth:"
        )
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        try:
            # Generate Hypothetical Ground Truths
            outputs = self.model.generate(
                inputs,
                max_length=self.max_length,
                num_return_sequences=self.num_ground_truths,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode the generated texts
            ground_truths = [
                self.tokenizer.decode(output, skip_special_tokens=True).split("Hypothetical Ground Truth:")[1].strip()
                for output in outputs
            ]
            
            # Validate and sanitize ground truths
            if self.privacy_sanitizer:
                validated_ground_truths = [self.privacy_sanitizer.validate_ground_truth(gt) for gt in ground_truths]
                self.logger.debug("Generated ground truths have been validated and sanitized.")
            else:
                validated_ground_truths = ground_truths
            
            # Calculate confidence scores
            confidence_scores = [self._calculate_confidence(gt) for gt in validated_ground_truths]
            
            # Check existing data in the database and compare confidence
            for gt, score in zip(validated_ground_truths, confidence_scores):
                existing_profile = self.db.get_user_profile()
                if existing_profile:
                    existing_score = self._calculate_confidence(existing_profile)
                    if score > existing_score:
                        self.db.update_user_profile(gt)
                        self.logger.debug(f"User profile updated with new ground truth (confidence: {score}).")
                    else:
                        self.logger.debug(f"Existing profile retained (higher confidence: {existing_score}).")
                else:
                    if score > self.config['confidence_threshold']:
                        self.db.update_user_profile(gt)
                        self.logger.debug(f"New user profile created with ground truth (confidence: {score}).")
                    else:
                        self.logger.debug(f"Inconclusive data check (confidence: {score}), no update made.")
            
            # Update cache with new ground truths
            if self.cache:
                self.cache.set(cache_key, validated_ground_truths)
                self.logger.info("Cache updated with new ground truths.")
            
            # Update user profile in the database
            for gt in validated_ground_truths:
                self.db.update_user_profile(gt)
                self.logger.debug("User profile updated in the database.")
            
            self.logger.info("Generated Hypothetical Ground Truths successfully.")
            return validated_ground_truths
        
        except Exception as e:
            self.logger.error(f"Error during ground truth generation: {e}")
            return []

    def _calculate_confidence(self, ground_truth: str) -> float:
        """
        Calculate confidence score for the generated ground truth.
        The score is based on how often similar conclusions were drawn in the past.
        :param ground_truth: The ground truth text.
        :return: Confidence score (0.0 to 1.0).
        """
        # Example: Count how many times a similar ground truth has been generated
        past_ground_truths = self.db.get_past_ground_truths()  # Hypothetical DB method to retrieve past conclusions
        similar_count = sum(1 for past_gt in past_ground_truths if self._is_similar(ground_truth, past_gt))
        
        # Confidence is the ratio of similar conclusions to the total past conclusions
        total_count = len(past_ground_truths)
        confidence_score = similar_count / total_count if total_count > 0 else 0.5  # Default to 0.5 for new profiles
        
        return confidence_score

    def _is_similar(self, gt1: str, gt2: str) -> bool:
        """
        Check if two ground truths are similar using cosine similarity between their embeddings.
        :param gt1: Ground truth 1.
        :param gt2: Ground truth 2.
        :return: Boolean indicating whether they are similar.
        """
        # Tokenize and generate embeddings for both ground truths
        gt1_embedding = self._get_embedding(gt1)
        gt2_embedding = self._get_embedding(gt2)
        
        # Calculate cosine similarity between the embeddings
        similarity_score = cosine_similarity(gt1_embedding, gt2_embedding)[0][0]
        
        # Define a threshold for similarity (e.g., 0.85 is a common threshold)
        similarity_threshold = self.config.get('similarity_threshold', 0.85)
        
        return similarity_score >= similarity_threshold

    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Generate the embedding for a given text using the model's hidden states.
        :param text: Input text to generate the embedding for.
        :return: Embedding tensor for the text.
        """
        # Tokenize the input text
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Pass the input through the model to get hidden states
        with torch.no_grad():
            outputs = self.model(inputs, output_hidden_states=True)
        
        # Get the last hidden state (embedding of the sequence)
        hidden_states = outputs.hidden_states[-1]  # Last layer's hidden states
        
        # Average the hidden states across all tokens to get a single vector representation
        embedding = hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
        # Convert to numpy for cosine similarity
        return embedding.cpu().numpy()

    def _generate_cache_key(self, context: str) -> str:
        """
        Generate a unique cache key based on the context.
        :param context: Combined conversation history, trends, and retrieved documents.
        :return: Cache key string.
        """
        import hashlib
        return hashlib.sha256(context.encode('utf-8')).hexdigest()
