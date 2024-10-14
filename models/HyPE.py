# models/hyde.py
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

class HyDE:
    def __init__(self, model_name="NousResearch/Llama-3.2-1B", device=torch.device("cpu"), max_length=200, num_ground_truths=1):
        """
        Initialize the HyDE module with a language model and tokenizer.
        :param model_name: Pretrained language model name or path.
        :param device: torch.device (cpu or cuda)
        :param max_length: Maximum length of generated Hypothetical Ground Truths.
        :param num_ground_truths: Number of Hypothetical Ground Truths to generate.
        """
        self.device = device
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.num_ground_truths = num_ground_truths

    def generate_hypothetical_ground_truths(self, conversation_history, retrieved_docs):
        """
        Generate Hypothetical Ground Truths based on conversation history and retrieved documents.
        :param conversation_history: List of past conversation strings.
        :param retrieved_docs: List of retrieved document strings.
        :return: List of generated Hypothetical Ground Truths.
        """
        # Combine conversation history and retrieved documents to form the context
        context = "\n".join(conversation_history) + "\n" + "\n".join(retrieved_docs)
        
        # Define the prompt for generating Hypothetical Ground Truths
        prompt = (
            f"You are an intelligent assistant analyzing the following conversation history and retrieved documents to uncover personal information.\n\n"
            f"Conversation History:\n{context}\n\n"
            f"Based on the above, generate a detailed Hypothetical Ground Truth that summarizes the user's profile, interests, and relevant personal information to enhance future responses.\n\n"
            f"Hypothetical Ground Truth:"
        )
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate Hypothetical Ground Truths
        outputs = self.model.generate(
            inputs,
            max_length=self.max_length,
            num_return_sequences=self.num_ground_truths,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode the generated texts
        ground_truths = [
            self.tokenizer.decode(output, skip_special_tokens=True).split("Hypothetical Ground Truth:")[1].strip()
            for output in outputs
        ]
        
        return ground_truths
