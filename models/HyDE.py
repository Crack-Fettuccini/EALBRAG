# models/hyde.py
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

class HyDE:
    def __init__(self, model, tokenizer, device, max_length=512, num_documents=1):
        """
        Initialize the HyDE module with a language model and tokenizer.
        :param model: Pretrained language model for generation
        :param tokenizer: Tokenizer corresponding to the model
        :param device: torch.device (cpu or cuda)
        :param max_length: Maximum length of generated hypothetical documents
        :param num_documents: Number of hypothetical documents to generate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.num_documents = num_documents
    
    def generate_hypothetical_documents(self, query):
        """
        Generate hypothetical documents based on the query.
        :param query: Input query string
        :return: List of generated hypothetical document strings
        """
        # Prepare the prompt for generating hypothetical documents
        prompt = f"Generate a detailed document relevant to the following query: {query}\n\nDocument:"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate documents
        outputs = self.model.generate(
            inputs,
            max_length=self.max_length,
            num_return_sequences=self.num_documents,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode the generated documents
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True).split("Document:")[1].strip() for output in outputs]
        
        return generated_texts
