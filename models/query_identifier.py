import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define parameters
max_window_size = 2048  # Model's max token limit for sliding window
overlap_size = 200      # Overlap to maintain context between chunks
tfidf_threshold = 0.2   # Threshold for considering high-importance terms

# Load corpus
def load_corpus(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Split corpus into overlapping chunks
def split_into_chunks(corpus, max_window_size, overlap_size):
    tokens = tokenizer.tokenize(corpus)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_window_size
        chunk = tokens[start:end]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
        start += max_window_size - overlap_size  # Move window with overlap
    return chunks

# Extract keywords using TF-IDF for each chunk
def extract_keywords(chunks):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = vectorizer.fit_transform(chunks)
    terms = vectorizer.get_feature_names_out()
    keywords = {}

    for i, chunk in enumerate(chunks):
        tfidf_scores = tfidf_matrix[i].toarray().flatten()
        high_tfidf_indices = np.where(tfidf_scores > tfidf_threshold)[0]
        chunk_keywords = [terms[idx] for idx in high_tfidf_indices]
        
        for word in chunk_keywords:
            if word in keywords:
                keywords[word] += tfidf_scores[terms.tolist().index(word)]
            else:
                keywords[word] = tfidf_scores[terms.tolist().index(word)]
                
    return keywords

# Identify primary and secondary queries
def identify_queries(keywords, primary_threshold=0.7, secondary_threshold=0.4):
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    
    primary_queries = [word for word, score in sorted_keywords if score > primary_threshold]
    secondary_queries = [word for word, score in sorted_keywords if secondary_threshold < score <= primary_threshold]
    
    return primary_queries, secondary_queries

# Main function
def main(file_path):
    corpus = load_corpus(file_path)
    chunks = split_into_chunks(corpus, max_window_size, overlap_size)
    
    # Extract keywords across all chunks
    keywords = extract_keywords(chunks)
    
    # Identify primary and secondary queries
    primary_queries, secondary_queries = identify_queries(keywords)
    
    print("Primary Queries:", primary_queries)
    print("Secondary Queries:", secondary_queries)

# Run the main function with an example corpus file
if __name__ == "__main__":
    main("path/to/your/corpus.txt")
