import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.memory_attention import MemoryAttention
from collections import deque
from HyPE import HyPE
import logging
from typing import Tuple, List, Optional

# Configure structured logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenGeneratorWithAttention:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8", 
                 max_window_size: int = 2048, 
                 attention_threshold: float = 0.1, 
                 memory_size: int = 1000, 
                 device: Optional[torch.device] = None,
                 gradient_checkpointing: bool = False):
        """
        Initializes the TokenGeneratorWithAttention class.
        
        Args:
            model_name (str): Pre-trained model to load.
            max_window_size (int): Maximum token context window.
            attention_threshold (float): Threshold for identifying important tokens.
            memory_size (int): Size of the memory storage for important tokens.
            device (torch.device, optional): Device for computation, defaults to GPU if available.
            gradient_checkpointing (bool): Enable gradient checkpointing for memory optimization.
        """
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        logger.info(f"Using device: {self.device}")

        # Load the tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                output_attentions=True, 
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            ).to(self.device)
            
            # Optionally enable gradient checkpointing for memory efficiency
            if gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing for memory optimization.")

            logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise RuntimeError("Model initialization failed.") from e

        # Initialize MemoryAttention for enhanced attention
        try:
            self.memory_attention = MemoryAttention(
                embed_dim=self.model.config.hidden_size,
                num_heads=self.model.config.num_attention_heads,
                memory_size=memory_size,
                lstm_hidden_size=256,
                num_lstm_layers=1,
                dropout=0.1
            )
            logger.info("MemoryAttention initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryAttention: {e}")
            raise

        # Initialize other parameters
        self.max_window_size = max_window_size
        self.attention_threshold = attention_threshold
        self.memory = deque(maxlen=memory_size)
        self.hype = HyPE()

    def track_attention(self, attentions: List[torch.Tensor], input_ids: torch.Tensor) -> List[int]:
        """
        Tracks and extracts important tokens based on attention scores.

        Args:
            attentions (List[torch.Tensor]): Attention scores from the model.
            input_ids (torch.Tensor): Current input token IDs.
        
        Returns:
            List[int]: Token indices marked as important.
        """
        try:
            important_tokens = set()
            for layer_attention in attentions:
                avg_attention = layer_attention.mean(dim=1)  # Average over attention heads
                important_tokens.update(
                    (avg_attention > self.attention_threshold).nonzero(as_tuple=True)[1].tolist()
                )
            return list(important_tokens)
        except Exception as e:
            logger.warning(f"Error during attention tracking: {e}")
            return []

    @torch.inference_mode()
    def generate_tokens(self, input_prompt: str, max_length: int = 100, verbose: bool = False) -> Tuple[str, List[str]]:
        """
        Generates text using attention-aware token tracking and memory management.
        
        Args:
            input_prompt (str): Starting prompt for the model.
            max_length (int): Maximum number of tokens to generate.
            verbose (bool): If True, logs each token generated.
        
        Returns:
            Tuple[str, List[str]]: Generated text and hypothetical ground truths.
        """
        try:
            input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to(self.device)
            attention_mask = torch.ones_like(input_ids, device=self.device)
            generated = input_ids.clone()

            for _ in range(max_length):
                outputs = self.model(input_ids=generated, attention_mask=attention_mask)
                logits = outputs.logits
                attentions = outputs.attentions

                important_tokens = self.track_attention(attentions, generated)
                self.memory.extend(tok for tok in important_tokens if tok not in self.memory)

                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated = torch.cat([generated, next_token_id], dim=1)

                # Sliding window logic to maintain max window size
                if generated.size(1) > self.max_window_size:
                    generated = generated[:, -self.max_window_size:]

                attention_mask = torch.ones_like(generated, device=self.device)

                if verbose:
                    logger.info(f"Generated Token: {self.tokenizer.decode(next_token_id)}")

                # Stop generation if EOS token is encountered
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

            response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            logger.info(f"Final response: {response}")

            # Hypothetical ground truths via HyPE
            conversation_history = [input_prompt]
            retrieved_docs = []
            ground_truths = self.hype.generate_hypothetical_ground_truths(conversation_history, retrieved_docs)

            return response, ground_truths
        except Exception as e:
            logger.error(f"Error during token generation: {e}")
            return "", []

    def save_model(self, save_path: str):
        """
        Saves the current model and tokenizer to the specified path.

        Args:
            save_path (str): Directory path to save the model.
        """
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model and tokenizer saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise IOError(f"Saving model failed: {e}")

    def load_model(self, load_path: str):
        """
        Loads the model and tokenizer from a specified directory.

        Args:
            load_path (str): Directory path to load the model and tokenizer.
        """
        try:
            self.model = AutoModelForCausalLM.from_pretrained(load_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            logger.info(f"Model and tokenizer loaded from {load_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {load_path}: {e}")
            raise IOError(f"Loading model failed: {e}")
