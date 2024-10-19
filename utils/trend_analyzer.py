import spacy
from collections import Counter
from transformers import LlamaTokenizer

class TrendAnalyzer:
    def __init__(self, config):
        """
        Initializes the trend analyzer with the Llama tokenizer and spaCy model.
        :param config: Configuration dictionary for trend analysis.
        """
        # Extract relevant config for TrendAnalyzer
        trend_analyzer_config = config.get('trend_analyzer', {})

        self.tokenizer = LlamaTokenizer.from_pretrained(config.get('llama_model_name', 'NousResearch/Llama-3.2-1B'))
        self.nlp = spacy.load(config.get('spacy_model', 'en_core_web_sm'))
        self.top_n_keywords = config.get('top_n_keywords', 10)
        self.entity_types = config.get('entity_types', ['PERSON', 'ORG', 'GPE', 'DATE', 'EVENT'])
    
    def analyze_trends(self, conversation_history: list) -> list:
        """
        Analyzes the trends in the given conversation history using Llama tokenizer for keywords
        and spaCy for named entity extraction.
        :param conversation_history: List of conversation strings.
        :return: List of key trends such as frequent tokens and entities.
        """
        combined_history = " ".join(conversation_history)
        
        # Tokenize conversation history using the Llama tokenizer
        tokens = self.tokenizer.tokenize(combined_history)
        
        # Extract named entities using spaCy
        doc = self.nlp(combined_history)
        entities = [ent.text for ent in doc.ents if ent.label_ in self.entity_types]
        
        # Count token frequencies (excluding special tokens)
        token_frequencies = Counter([token for token in tokens if not token.startswith('▁')]).most_common(self.top_n_keywords)
        
        # Count entity frequencies
        entity_frequencies = Counter(entities).most_common(self.top_n_keywords)
        
        # Combine token frequencies and entities as trends
        trends = [f"Token: {token} ({count})" for token, count in token_frequencies] + \
                 [f"Entity: {entity} ({count})" for entity, count in entity_frequencies]
        
        return trends
