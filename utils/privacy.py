import re
import logging
import yaml
import os

class PrivacySanitizer:
    """
    A class to handle sanitization of input texts based on configurable privacy rules.
    """

    def __init__(self, config_path: str):
        """
        Initialize the PrivacySanitizer by loading YAML configuration directly from a file.
        
        :param config_path: Path to the configuration YAML file.
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.patterns = self._load_patterns(self.config)
        self.key_path = self.config.get('local_key_path', None)
        self._compile_patterns()

    def _load_config(self, config_path: str):
        """
        Load the configuration from a YAML file.
        
        :param config_path: Path to the YAML file.
        :return: Configuration dictionary.
        """
        if not config_path or not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _load_patterns(self, config):
        """
        Load sanitization patterns from configuration.

        :param config: Configuration dictionary.
        :return: Dictionary mapping data types to their regex patterns.
        """
        return config.get('prohibited_data_types_patterns', {})

    def _compile_patterns(self):
        """
        Precompile regex patterns for performance.
        """
        self.compiled_patterns = {}
        for data_type, patterns in self.patterns.items():
            self.compiled_patterns[data_type] = []
            for pattern in patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                    self.compiled_patterns[data_type].append(compiled)
                except re.error as e:
                    self.logger.error(f"Invalid regex pattern for {data_type}: {pattern} | Error: {e}")
        self.logger.debug("All regex patterns compiled successfully.")

    def _is_local_access(self):
        """
        Check if local access is available using a local key for deserialization.
        :return: True if the local key is available, False otherwise.
        """
        if self.key_path and os.path.exists(self.key_path):
            self.logger.info("Local deserialization key found. Sanitization disabled for local access.")
            return True
        else:
            self.logger.info("No local deserialization key found. Sanitization enabled.")
            return False

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input text by redacting prohibited data types, if local key is unavailable.

        :param text: The input text to sanitize.
        :return: Sanitized text.
        """
        if not isinstance(text, str):
            self.logger.warning(f"Expected string input, got {type(text)}. Converting to string.")
            text = str(text)

        if self._is_local_access():
            # If local key is available, do not sanitize
            self.logger.info("Local key detected. Returning original text without sanitization.")
            return text

        # Otherwise, sanitize the text
        sanitized_text = text
        for data_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                sanitized_text, num_subs = pattern.subn(self._get_replacement(data_type), sanitized_text)
                if num_subs > 0:
                    self.logger.info(f"Redacted {num_subs} instance(s) of {data_type}.")
        return sanitized_text

    def validate_ground_truth(self, text: str) -> str:
        """
        Validate and sanitize the generated ground truth.

        :param text: Generated ground truth text.
        :return: Validated ground truth text.
        """
        return self.sanitize_input(text)

    def _get_replacement(self, data_type: str) -> str:
        """
        Get the replacement string based on data type.

        :param data_type: The type of data being redacted.
        :return: Replacement string.
        """
        replacement = self.config.get('replacement_strings', {}).get(data_type, '[REDACTED]')
        return replacement
