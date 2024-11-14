import unittest
import torch
from torch import nn
from torch.nn import functional as F
from torch.testing import assert_allclose
from unittest.mock import MagicMock

class TestDocumentReindexing(unittest.TestCase):

    def setUp(self):
        # Set up a dummy model and data for testing
        self.embed_dim = 64
        self.num_heads = 8
        self.batch_size = 2
        self.num_docs = 5
        self.doc_len = 10
        self.query_len = 4
        
        self.model = DocumentReindexing(embed_dim=self.embed_dim, num_heads=self.num_heads)
        
        # Mocking a dummy query and documents for input
        self.query = torch.rand(self.batch_size, self.query_len, self.embed_dim)
        self.documents = torch.rand(self.batch_size, self.num_docs, self.doc_len, self.embed_dim)
        
    def test_reorder_documents(self):
        """Test that the documents are reordered correctly"""
        reordered_docs = self.model.reorder_documents(self.query, self.documents)
        self.assertEqual(reordered_docs.shape, self.documents.shape)
    
    def test_compute_attention(self):
        """Test attention computation between query and documents"""
        attention_weights = self.model.compute_attention(self.query, self.documents)
        self.assertEqual(attention_weights.shape, (self.batch_size, self.num_docs, self.query_len))

    def test_apply_attention_reordering(self):
        """Test the application of attention reordering"""
        attention_weights = torch.rand(self.batch_size, self.num_docs, self.query_len)
        reordered_docs = self.model.apply_attention_reordering(self.documents, attention_weights)
        self.assertEqual(reordered_docs.shape, self.documents.shape)

    def test_chunk_documents(self):
        """Test the document chunking functionality"""
        chunk_sizes = [[3, 4, 3], [5, 5]]
        chunked_docs = self.model.chunk_documents(self.documents, chunk_sizes)
        self.assertEqual(chunked_docs.shape[0], self.batch_size)
        self.assertEqual(chunked_docs.shape[1], self.num_docs)
        
    def test_handle_multimodal_inputs(self):
        """Test handling multimodal inputs (text and image documents)"""
        text_documents = torch.rand(self.batch_size, self.num_docs, self.doc_len, self.embed_dim)
        image_documents = torch.rand(self.batch_size, self.num_docs, self.doc_len, self.embed_dim)
        
        reordered_text_docs, reordered_image_docs = self.model.handle_multimodal_inputs(self.query, text_documents, image_documents)
        
        self.assertEqual(reordered_text_docs.shape, text_documents.shape)
        self.assertEqual(reordered_image_docs.shape, image_documents.shape)


class TestDocumentReindexingWithDP(unittest.TestCase):
    def setUp(self):
        # Set up a dummy model and data for testing
        self.embed_dim = 64
        self.num_heads = 8
        self.batch_size = 2
        self.num_docs = 5
        self.doc_len = 10
        self.query_len = 4
        self.model = DocumentReindexingWithDP(embed_dim=self.embed_dim, num_heads=self.num_heads)
        
        # Mocking a dummy query and documents for input
        self.query = torch.rand(self.batch_size, self.query_len, self.embed_dim)
        self.documents = torch.rand(self.batch_size, self.num_docs, self.doc_len, self.embed_dim)
        self.chunk_texts = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5"]
        self.prompt_text = "This is a test prompt."
        
    def test_reorder_and_optimize(self):
        """Test that reordering and optimization of document chunks works"""
        # Mock the model and tokenizer
        model = MagicMock()
        tokenizer = MagicMock()
        
        tokenizer.return_value = {'input_ids': torch.randint(0, 1000, (1, 10))}
        
        # Mocking model's output for the next token probability
        model.return_value = MagicMock(logits=torch.rand(1, 10, 1000))
        
        reordered_docs, best_score = self.model.reorder_and_optimize(
            model=model,
            tokenizer=tokenizer,
            query=self.query,
            chunk_texts=self.chunk_texts,
            prompt_text=self.prompt_text,
            documents=self.documents,
            chunk_sizes=[[3, 4, 3], [5, 5]]
        )
        
        self.assertEqual(reordered_docs.shape, self.documents.shape)
        self.assertIsInstance(best_score, float)
        self.assertGreaterEqual(best_score, 0.0)
        
    def test_dp_reordering_with_attention_optimization(self):
        """Test that dynamic programming reordering with attention optimization works"""
        model = MagicMock()
        tokenizer = MagicMock()
        
        # Mocking the output of the tokenizer and model
        tokenizer.return_value = {'input_ids': torch.randint(0, 1000, (1, 10))}
        model.return_value = MagicMock(logits=torch.rand(1, 10, 1000))
        
        best_sequence, best_score = dp_reordering_with_attention_optimization(
            model=model,
            tokenizer=tokenizer,
            query=self.query,
            chunk_texts=self.chunk_texts,
            prompt_text=self.prompt_text
        )
        
        self.assertIsInstance(best_sequence, str)
        self.assertIsInstance(best_score, float)
        self.assertGreaterEqual(best_score, 0.0)


if __name__ == '__main__':
    unittest.main()
