import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from database.user_feedback_db import UserFeedbackDB  # Hypothetical feedback database
from database.pathway_weight_db import PathwayWeightDB  # Hypothetical pathway weight database
import yaml

class ReinforcementManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the ReinforcementManager with configurations, sentiment model, databases, and logging.
        :param config_path: Path to the configuration YAML file.
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set up logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize sentiment analysis model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.config['sentiment_model']['name']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['sentiment_model']['name'])

        # Initialize databases
        self.feedback_db = UserFeedbackDB(self.config['database']['feedback_connection_string'])
        self.weight_db = PathwayWeightDB(self.config['database']['pathway_connection_string'])
        
        self.logger.info("ReinforcementManager initialized successfully.")

    def setup_logging(self):
        """
        Setup logging based on configuration.
        """
        log_level_str = self.config.get('logging', {}).get('level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_file = self.config.get('logging', {}).get('log_file', 'reinforcement.log')
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def adjust_pathway_weights(self):
        """
        Iteratively adjust pathway weights based on user feedback sentiment.
        """
        while True:
            # Fetch recent conversation history for feedback
            recent_conversations = self.feedback_db.get_recent_conversations()
            if not recent_conversations:
                self.logger.info("No recent conversations to process.")
                time.sleep(self.config['reinforcement']['interval'])
                continue
            
            # Process each conversation history entry
            for conversation in recent_conversations:
                user_id = conversation['user_id']
                pathway_id = conversation['pathway_id']
                user_responses = conversation['responses']
                
                # Calculate sentiment score for user responses
                sentiment_score = self._analyze_sentiment(user_responses)
                
                # Classify feedback as positive or negative based on sentiment score
                if sentiment_score > self.config['reinforcement']['positive_threshold']:
                    satisfaction = 1  # Positive feedback
                elif sentiment_score < self.config['reinforcement']['negative_threshold']:
                    satisfaction = 0  # Negative feedback
                else:
                    satisfaction = 0.5  # Neutral or inconclusive feedback
                
                # Retrieve and adjust pathway weight based on classified satisfaction
                current_weight = self.weight_db.get_weight(pathway_id)
                new_weight = self._calculate_new_weight(current_weight, satisfaction)
                self.weight_db.update_weight(pathway_id, new_weight)
                
                self.logger.info(f"Updated weight for pathway {pathway_id} (User: {user_id}) to {new_weight}")

            # Sleep for a configured interval before checking for more feedback
            time.sleep(self.config['reinforcement']['interval'])

    def _analyze_sentiment(self, responses):
        """
        Analyze sentiment of user responses and return an aggregated sentiment score.
        :param responses: List of user response texts.
        :return: Aggregated sentiment score (0 to 1 scale).
        """
        # Tokenize and predict sentiment for each response
        total_score = 0
        for response in responses:
            inputs = self.tokenizer(response, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                positive_score = probabilities[0][1].item()  # Assuming index 1 is positive sentiment
                total_score += positive_score

        # Calculate the average sentiment score for the conversation
        return total_score / len(responses) if responses else 0.5  # Default to neutral if empty

    def _calculate_new_weight(self, current_weight: float, satisfaction: float) -> float:
        """
        Calculate a new weight for a pathway based on satisfaction feedback.
        :param current_weight: The current weight for the pathway.
        :param satisfaction: Satisfaction metric based on sentiment (0, 0.5, or 1).
        :return: Adjusted weight.
        """
        learning_rate = self.config['reinforcement']['learning_rate']
        
        # Adjust weight based on satisfaction
        adjustment = learning_rate * (satisfaction - 0.5)  # Positive or negative based on satisfaction
        new_weight = current_weight + adjustment
        
        # Ensure weight stays within bounds (0 to 1)
        return max(0.0, min(1.0, new_weight))

# Usage:
if __name__ == "__main__":
    # Initialize and run the ReinforcementManager
    reinforcement_manager = ReinforcementManager(config_path="config/config.yaml")
    reinforcement_manager.adjust_pathway_weights()
