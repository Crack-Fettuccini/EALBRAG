import unittest
from utils.privacy import PrivacySanitizer
import logging

class TestPrivacySanitizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure logging to capture logs for testing
        logging.basicConfig(level=logging.CRITICAL)  # Suppress logs during testing
        
        # Define test configuration
        cls.test_config = {
            'prohibited_data_types_patterns': {
                'sensitive_information': [
                    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
                ],
                'financial_data': [
                    r'\$\d+(\.\d{2})?'  # Financial amounts
                ]
            },
            'replacement_strings': {
                'sensitive_information': '[REDACTED SENSITIVE INFORMATION]',
                'financial_data': '[REDACTED FINANCIAL DATA]'
            }
        }
        cls.sanitizer = PrivacySanitizer(cls.test_config)
    
    def test_sanitize_ssn(self):
        input_text = "User's SSN is 123-45-6789."
        expected_output = "User's SSN is [REDACTED SENSITIVE INFORMATION]."
        sanitized = self.sanitizer.sanitize_input(input_text)
        self.assertEqual(sanitized, expected_output)
    
    def test_sanitize_email(self):
        input_text = "Contact me at user@example.com."
        expected_output = "Contact me at [REDACTED SENSITIVE INFORMATION]."
        sanitized = self.sanitizer.sanitize_input(input_text)
        self.assertEqual(sanitized, expected_output)
    
    def test_sanitize_financial_data(self):
        input_text = "I have $1000 in my account."
        expected_output = "I have [REDACTED FINANCIAL DATA] in my account."
        sanitized = self.sanitizer.sanitize_input(input_text)
        self.assertEqual(sanitized, expected_output)
    
    def test_multiple_sanitizations(self):
        input_text = "My SSN is 123-45-6789 and my email is user@example.com. I owe $500."
        expected_output = (
            "My SSN is [REDACTED SENSITIVE INFORMATION] and my email is [REDACTED SENSITIVE INFORMATION]. "
            "I owe [REDACTED FINANCIAL DATA]."
        )
        sanitized = self.sanitizer.sanitize_input(input_text)
        self.assertEqual(sanitized, expected_output)
    
    def test_no_sanitization_needed(self):
        input_text = "I enjoy hiking and outdoor activities."
        expected_output = "I enjoy hiking and outdoor activities."
        sanitized = self.sanitizer.sanitize_input(input_text)
        self.assertEqual(sanitized, expected_output)
    
    def test_non_string_input(self):
        input_text = 12345  # Non-string input
        expected_output = "12345"
        sanitized = self.sanitizer.sanitize_input(input_text)
        self.assertEqual(sanitized, expected_output)
    
    def test_invalid_regex_pattern(self):
        # Adding an invalid regex pattern to the configuration
        invalid_config = {
            'prohibited_data_types_patterns': {
                'test_type': [
                    r'['  # Invalid regex
                ]
            },
            'replacement_strings': {
                'test_type': '[REDACTED TEST]'
            }
        }
        sanitizer = PrivacySanitizer(invalid_config)
        # The invalid pattern should be skipped, no redaction occurs
        input_text = "This is a test [example]."
        expected_output = "This is a test [example]."
        sanitized = sanitizer.sanitize_input(input_text)
        self.assertEqual(sanitized, expected_output)

if __name__ == '__main__':
    unittest.main()
