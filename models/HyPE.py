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
        :param conversation_history: List of past conversation strings.
        :param retrieved_docs: List of retrieved document strings.
        :return: List of generated Hypothetical Ground Truths.
        """
        # Sanitize inputs for privacy
        if self.privacy_sanitizer:
            sanitized_history = [self.privacy_sanitizer.sanitize_input(text) for text in conversation_history]
            sanitized_docs = [self.privacy_sanitizer.sanitize_input(doc) for doc in retrieved_docs]
            self.logger.debug("Input texts have been sanitized.")
        else:
            sanitized_history = conversation_history
            sanitized_docs = retrieved_docs
        
        # Summarize conversation history to handle large contexts
        summarized_history = self.summarizer.summarize_conversation(sanitized_history)
        self.logger.debug("Conversation history summarized.")
        
        # Perform trend analysis to capture key trends in conversation history
        trends = self.trend_analyzer.analyze_trends(sanitized_history)
        self.logger.debug("Trend analysis complete.")
        
        # Combine summarized history, trends, and retrieved documents to form the context
        context = "\n".join(summarized_history) + "\n" + "\n".join(sanitized_docs) + "\n" + "\n".join(trends)
        
        # Check cache
        cache_key = self._generate_cache_key(context)
        if self.cache and self.cache.contains(cache_key):
            self.logger.info("Cache hit for the given context.")
            return self.cache.get(cache_key)
        
        # Define the prompt for generating Hypothetical Ground Truths
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
            
            # Update cache
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
    
    def _generate_cache_key(self, context: str) -> str:
        """
        Generate a unique cache key based on the context.
        :param context: Combined conversation history, trends, and retrieved documents.
        :return: Cache key string.
        """
        import hashlib
        return hashlib.sha256(context.encode('utf-8')).hexdigest()
