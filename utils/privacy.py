import re
import logging
from typing import Dict, List, Pattern, Any
import yaml
import os

class PrivacySanitizer:
    """
    A class to handle sanitization of input texts based on configurable privacy rules.
    Sanitization is only applied if the data is non-local (i.e., key is not present on the device).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PrivacySanitizer with given configuration and key.
        The key is embedded within the device for local-only processing.
        
        :param config: Dictionary containing privacy settings.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.patterns = self._load_patterns(config)
        
        # Hardcoding the path to the local key on the device
        self.key_path = '/secure/local/device_key'  # Local key embedded in the device
        self._compile_patterns()

    def _load_patterns(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
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
        self.compiled_patterns: Dict[str, List[Pattern]] = {}
        for data_type, patterns in self.patterns.items():
            self.compiled_patterns[data_type] = []
            for pattern in patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                    self.compiled_patterns[data_type].append(compiled)
                except re.error as e:
                    self.logger.error(f"Invalid regex pattern for {data_type}: {pattern} | Error: {e}")
        self.logger.debug("All regex patterns compiled successfully.")

    def is_local(self) -> bool:
        """
        Check if the processing is happening on a local device by verifying the presence of the embedded key.
        
        :return: True if the key exists (indicating local processing), False otherwise.
        """
        # Check if the local key exists on the device
        if os.path.exists(self.key_path):
            self.logger.info(f"Local key found at {self.key_path}. Processing locally.")
            return True
        else:
            self.logger.warning(f"Local key not found at {self.key_path}. Sanitization will be applied.")
            return False

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input text by redacting prohibited data types only if non-local processing is detected.

        :param text: The input text to sanitize.
        :return: Sanitized text (if non-local) or original text (if local).
        """
        if not isinstance(text, str):
            self.logger.warning(f"Expected string input, got {type(text)}. Converting to string.")
            text = str(text)

        # Only sanitize if not local
        if self.is_local():
            self.logger.info("Processing locally, no sanitization applied.")
            return text  # Do not sanitize if processing locally.

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
        :return: Validated ground truth text (sanitized if non-local).
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
