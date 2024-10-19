from transformers import BartForConditionalGeneration, BartTokenizer
import torch

class Summarizer:
    def __init__(self, config):
        """
        Initializes the summarizer model.
        :param config: Configuration dictionary for summarization.
        """
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_name = config.get('model_name', 'facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.max_length = config.get('max_length', 512)
        self.summary_length = config.get('summary_length', 150)
        
    def summarize_conversation(self, conversation_history: list) -> list:
        """
        Summarizes the given conversation history.
        :param conversation_history: List of conversation strings.
        :return: Summarized version of the conversation as a list of key points.
        """
        combined_history = " ".join(conversation_history)
        
        # Tokenize input
        inputs = self.tokenizer.encode(combined_history, return_tensors='pt', max_length=self.max_length, truncation=True).to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(inputs, max_length=self.summary_length, num_beams=4, length_penalty=2.0, early_stopping=True)
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return [summary]

