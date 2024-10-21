import torch
from transformers import AutoTokenizer, LlamaForCausalLM

class HyDE:
    def __init__(self, model, tokenizer, device, max_length=200, num_documents=1, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.0):
        """
        Initialize the HyDE module with a language model and tokenizer.
        :param model: Pretrained language model for generation
        :param tokenizer: Tokenizer corresponding to the model
        :param device: torch.device (cpu or cuda)
        :param max_length: Maximum length of generated hypothetical documents
        :param num_documents: Number of hypothetical documents to generate
        :param temperature: Sampling temperature for generation diversity
        :param top_p: Nucleus sampling parameter
        :param top_k: Top-k sampling parameter
        :param repetition_penalty: Penalty for repeated phrases to enhance diversity
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

    def generate_hypothetical_documents(self, query, additional_context=None):
        """
        Generate hypothetical documents based on the query.
        :param query: Input query string
        :param additional_context: Optional string to provide more context for generation
        :return: List of generated hypothetical document strings
        """
        # Prepare the prompt, with optional additional context
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
        
        # Decode the generated documents and clean them
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True).split("Document:")[1].strip() 
            for output in outputs
        ]
        
        # Remove any empty or incomplete documents
        generated_texts = [text for text in generated_texts if len(text) > 0]
        
        return generated_texts

    def generate_and_return_token_ids(self, query):
        """
        Generate hypothetical documents and return both text and token IDs for further processing.
        :param query: Input query string
        :return: List of tuples containing (generated_text, token_ids)
        """
        prompt = f"Generate a detailed document relevant to the following query: {query}\n\nDocument:"
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
        
        # Decode and return both token IDs and texts
        results = [
            (self.tokenizer.decode(output, skip_special_tokens=True).split("Document:")[1].strip(), output)
            for output in outputs
        ]
        
        return results
