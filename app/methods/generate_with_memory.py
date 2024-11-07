import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.memory_attention import MemoryAttention  # Ensure this path is correct
from collections import deque
from HyPE import HyPE  # Ensure correct import
import logging
from typing import Tuple, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenGeneratorWithAttention:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8", 
                 max_window_size: int = 2048, 
                 attention_threshold: float = 0.1, 
                 memory_size: int = 1000,
                 device: Optional[torch.device] = None):
        """
        Initializes TokenGeneratorWithAttention.
        Args:
            model_name (str): Name of the pre-trained model.
            max_window_size (int): Maximum sliding window size.
            attention_threshold (float): Threshold to select important tokens.
            memory_size (int): Maximum size of memory storage.
            device (torch.device, optional): Device for computation (CPU/GPU).
        """
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        logger.info(f"Using device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                output_attentions=True, 
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            ).to(self.device)

            logger.info(f"Loaded model {model_name} successfully.")

        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise

        # Initialize MemoryAttention
        self.memory_attention = MemoryAttention(
            embed_dim=self.model.config.hidden_size,
            num_heads=self.model.config.num_attention_heads,
            memory_size=memory_size,
            lstm_hidden_size=256,
            num_lstm_layers=1,
            dropout=0.1
        )

        # Initialize parameters
        self.max_window_size = max_window_size
        self.attention_threshold = attention_threshold
        self.memory = deque(maxlen=memory_size)  # Memory storage
        self.hype = HyPE()  # Hypothetical grounding system

    def track_attention(self, attentions: List[torch.Tensor], input_ids: torch.Tensor) -> List[int]:
        """
        Track important tokens based on attention scores.
        Args:
            attentions (List[torch.Tensor]): Attention scores from the model.
            input_ids (torch.Tensor): Tokenized input IDs.
        Returns:
            List[int]: List of important token IDs.
        """
        important_tokens = set()
        for layer_attention in attentions:
            avg_attention = layer_attention.mean(dim=1)  # Average across heads
            important_tokens.update((avg_attention > self.attention_threshold).nonzero(as_tuple=True)[1].tolist())
        return list(important_tokens)

    @torch.inference_mode()
    def generate_tokens(self, input_prompt: str, max_length: int = 100, verbose: bool = False) -> Tuple[str, List[str]]:
        """
        Generate tokens using attention-aware memory and sliding window.
        Args:
            input_prompt (str): Initial prompt for generation.
            max_length (int): Max number of tokens to generate.
            verbose (bool): Flag for verbose logging.
        Returns:
            Tuple[str, List[str]]: Generated text and HyPE ground truths.
        """
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

            # Handle sliding window
            if generated.size(1) > self.max_window_size:
                generated = generated[:, -self.max_window_size:]

            attention_mask = torch.ones_like(generated, device=self.device)

            if verbose:
                logger.info(f"Generated Token: {self.tokenizer.decode(next_token_id)}")

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        # Final response
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        # Generate hypothetical ground truths via HyPE
        conversation_history = [input_prompt]  # Example history, replace with actual
        retrieved_docs = []  # Add retrieved documents if any
        ground_truths = self.hype.generate_hypothetical_ground_truths(conversation_history, retrieved_docs)

        return response, ground_truths

    def save_model(self, save_path: str):
        """Save the current model state."""
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
