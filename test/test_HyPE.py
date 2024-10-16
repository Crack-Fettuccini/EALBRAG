import unittest
from models.HyPE import HyPE
import os

class TestHyPE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a test configuration file
        cls.test_config_path = "config/test_config.yaml"
        test_config = {
            'model': {
                'name': "NousResearch/Llama-3.2-1B",
                'device': "cpu",
                'max_length': 100,
                'num_ground_truths': 1,
                'temperature': 0.7,
                'top_p': 0.9
            },
            'privacy': {
                'enabled': True,
                'allowed_data_types': ['name', 'interests'],
                'prohibited_data_types': ['sensitive_information']
            },
            'cache': {
                'enabled': False,
                'cache_dir': "./cache/",
                'max_cache_size': 1000
            },
            'logging': {
                'level': "CRITICAL",
                'log_file': "test_hype.log"
            },
            'database': {
                'type': "sqlite",
                'connection_string': "sqlite:///./test_user_profiles.db"
            }
        }
        import yaml
        with open(cls.test_config_path, 'w') as file:
            yaml.dump(test_config, file)
        
        # Initialize HyPE
        cls.hype = HyPE(config_path=cls.test_config_path)
    
    @classmethod
    def tearDownClass(cls):
        # Remove test configuration and database
        os.remove(cls.test_config_path)
        if os.path.exists("test_user_profiles.db"):
            os.remove("test_user_profiles.db")
        if os.path.exists("test_hype.log"):
            os.remove("test_hype.log")
    
    def test_generate_hypothetical_ground_truths(self):
        conversation_history = [
            "User: I'm interested in hiking and outdoor activities.",
            "Assistant: That's great! Do you have any preferred brands for outdoor gear?"
        ]
        retrieved_docs = [
            "Document 1: Top hiking gear brands in 2024.",
            "Document 2: Essential equipment for outdoor enthusiasts."
        ]
        ground_truths = self.hype.generate_hypothetical_ground_truths(conversation_history, retrieved_docs)
        self.assertIsInstance(ground_truths, list)
        self.assertEqual(len(ground_truths), 1)
        self.assertIsInstance(ground_truths[0], str)
        self.assertIn("hiking", ground_truths[0].lower())
    
    def test_privacy_sanitization(self):
        sensitive_conversation = [
            "User: My SSN is 123-45-6789 and my email is user@example.com."
        ]
        retrieved_docs = []
        ground_truths = self.hype.generate_hypothetical_ground_truths(sensitive_conversation, retrieved_docs)
        self.assertNotIn("123-45-6789", ground_truths[0])
        self.assertNotIn("user@example.com", ground_truths[0])
        self.assertIn("[REDACTED SSN]", ground_truths[0])
        self.assertIn("[REDACTED EMAIL]", ground_truths[0])

if __name__ == '__main__':
    unittest.main()
