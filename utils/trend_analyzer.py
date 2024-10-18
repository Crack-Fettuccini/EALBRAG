import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class TrendAnalyzer:
    def __init__(self, config):
        """
        Initializes the trend analyzer.
        :param config: Configuration dictionary for trend analysis.
        """
        self.nlp = spacy.load(config.get('spacy_model', 'en_core_web_sm'))
        self.top_n_keywords = config.get('top_n_keywords', 10)
        self.entity_types = config.get('entity_types', ['PERSON', 'ORG', 'GPE', 'DATE', 'EVENT'])
    
    def analyze_trends(self, conversation_history: list) -> list:
        """
        Analyzes the trends in the given conversation history.
        :param conversation_history: List of conversation strings.
        :return: List of key trends such as frequent entities and topics.
        """
        combined_history = " ".join(conversation_history)
        
        # Extract named entities
        doc = self.nlp(combined_history)
        entities = [ent.text for ent in doc.ents if ent.label_ in self.entity_types]
        
        # Extract keywords using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=self.top_n_keywords)
        X = vectorizer.fit_transform([combined_history])
        keywords = vectorizer.get_feature_names_out()
        
        # Count entity frequencies
        entity_frequencies = Counter(entities).most_common(self.top_n_keywords)
        
        # Combine entities and keywords as trends
        trends = [f"Keyword: {keyword}" for keyword in keywords] + [f"Entity: {entity} ({count})" for entity, count in entity_frequencies]
        
        return trends

