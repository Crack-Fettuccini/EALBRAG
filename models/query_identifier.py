import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Load the model and tokenizer for semantic similarity
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load the corpus
def load_corpus(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Split corpus into sentences
def split_into_sentences(corpus):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', corpus)
    return [sentence.strip() for sentence in sentences if sentence]

# Detect possible queries based on sentence structure
def detect_possible_queries(sentences):
    query_phrases = ["what", "how", "does", "is", "are", "explain", "discuss"]
    possible_queries = []

    for sentence in sentences:
        # Identify potential questions/statements based on interrogative structure
        if any(sentence.lower().startswith(phrase) for phrase in query_phrases):
            possible_queries.append(sentence)
    
    return possible_queries

# Generate embeddings for sentences
def generate_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings

# Separate main content from identified queries
def identify_content_and_queries(sentences, possible_queries):
    # Embeddings for all sentences
    sentence_embeddings = generate_embeddings(sentences, tokenizer, model)
    
    # Detect main content: sentences dissimilar to identified queries
    main_content = []
    for i, sentence in enumerate(sentences):
        if sentence not in possible_queries:
            main_content.append(sentence)
    
    main_content_text = " ".join(main_content)
    return main_content_text, possible_queries

# Identify primary and secondary queries based on similarity to main content
def categorize_queries(main_content, queries, primary_threshold=0.75, secondary_threshold=0.5):
    main_embedding = generate_embeddings([main_content], tokenizer, model)
    query_embeddings = generate_embeddings(queries, tokenizer, model)

    # Calculate cosine similarity between main content and each query
    similarities = cosine_similarity(main_embedding, query_embeddings).flatten()
    
    # Categorize queries based on similarity thresholds
    primary_queries = [queries[i] for i, score in enumerate(similarities) if score >= primary_threshold]
    secondary_queries = [queries[i] for i, score in enumerate(similarities) if secondary_threshold <= score < primary_threshold]
    
    return primary_queries, secondary_queries

# Main function
def main(file_path):
    corpus = load_corpus(file_path)
    sentences = split_into_sentences(corpus)
    possible_queries = detect_possible_queries(sentences)
    
    if not possible_queries:
        print("No queries found in the corpus.")
        return

    # Identify main content and possible queries
    main_content, queries = identify_content_and_queries(sentences, possible_queries)
    
    # Categorize queries based on relevance to the main content
    primary_queries, secondary_queries = categorize_queries(main_content, queries)
    
    print("Primary Queries:")
    for query in primary_queries:
        print(f"- {query}")
    
    print("\nSecondary Queries:")
    for query in secondary_queries:
        print(f"- {query}")

# Run the main function with an example corpus file
#if __name__ == "__main__":
#    main("path/to/your/corpus.txt")
