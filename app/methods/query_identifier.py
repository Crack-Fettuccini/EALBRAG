import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re

class QueryIdentifier:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the QueryIdentifier with a specified model.
        :param model_name: Name of the pre-trained model to be used for embeddings.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def load_corpus(self, file_path):
        """
        Load the corpus from a specified file path.
        :param file_path: Path to the corpus file.
        :return: The content of the corpus as a string.
        """
        with open(file_path, 'r') as file:
            return file.read()

    def split_into_sentences(self, corpus):
        """
        Split the corpus into sentences.
        :param corpus: The complete corpus as a string.
        :return: A list of sentences.
        """
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', corpus)
        return [sentence.strip() for sentence in sentences if sentence]

    def detect_possible_queries(self, sentences):
        """
        Detect possible queries based on sentence structure.
        :param sentences: List of sentences from the corpus.
        :return: A list of detected queries.
        """
        query_phrases = ["what", "how", "does", "is", "are", "explain", "discuss"]
        possible_queries = []

        for sentence in sentences:
            if any(sentence.lower().startswith(phrase) for phrase in query_phrases):
                possible_queries.append(sentence)
        
        return possible_queries

    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        :param texts: List of strings to generate embeddings for.
        :return: Tensor of embeddings.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings

    def identify_content_and_queries(self, sentences, possible_queries):
        """
        Separate the main content from identified queries.
        :param sentences: List of all sentences.
        :param possible_queries: List of identified queries.
        :return: Main content text and possible queries.
        """
        main_content = []
        for sentence in sentences:
            if sentence not in possible_queries:
                main_content.append(sentence)
        
        main_content_text = " ".join(main_content)
        return main_content_text, possible_queries

    def categorize_queries(self, main_content, queries, primary_threshold=0.75, secondary_threshold=0.5):
        """
        Categorize queries based on similarity to the main content.
        :param main_content: The main content as a string.
        :param queries: List of possible queries.
        :param primary_threshold: Threshold for primary queries.
        :param secondary_threshold: Threshold for secondary queries.
        :return: Lists of primary and secondary queries.
        """
        main_embedding = self.generate_embeddings([main_content])
        query_embeddings = self.generate_embeddings(queries)

        similarities = cosine_similarity(main_embedding.numpy(), query_embeddings.numpy()).flatten()
        
        primary_queries = [queries[i] for i, score in enumerate(similarities) if score >= primary_threshold]
        secondary_queries = [queries[i] for i, score in enumerate(similarities) if secondary_threshold <= score < primary_threshold]
        
        return primary_queries, secondary_queries

    def process_corpus(self, file_path):
        """
        Process the corpus to identify and categorize queries.
        :param file_path: Path to the corpus file.
        """
        corpus = self.load_corpus(file_path)
        sentences = self.split_into_sentences(corpus)
        possible_queries = self.detect_possible_queries(sentences)
        
        if not possible_queries:
            print("No queries found in the corpus.")
            return

        main_content, queries = self.identify_content_and_queries(sentences, possible_queries)
        primary_queries, secondary_queries = self.categorize_queries(main_content, queries)
        
        print("Primary Queries:")
        for query in primary_queries:
            print(f"- {query}")
        
        print("\nSecondary Queries:")
        for query in secondary_queries:
            print(f"- {query}")

# Example usage
if __name__ == "__main__":
    query_identifier = QueryIdentifier()
    query_identifier.process_corpus("path/to/your/corpus.txt")
