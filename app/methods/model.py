import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class PassageCoverageAnalyzer:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)

    def analyze_coverage(self, input_text, threshold=0.1):
        """
        Analyzes cumulative attention weights to determine if all tokens in the passage
        were attended to sufficiently.

        Parameters:
            input_text (str): The input text to analyze.
            threshold (float): The minimum cumulative attention score for a token to be considered "attended."

        Returns:
            dict: A dictionary with cumulative attention scores and a flag indicating sufficient coverage.
        """
        # Tokenize the input and prepare for model
        inputs = self.tokenizer(input_text, return_tensors="pt")
        cumulative_attention_scores = torch.zeros(inputs["input_ids"].shape[-1])

        # Run the model and get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Aggregate attention across all layers and heads
        for layer_attention in outputs.attentions:
            # Average over heads and sum across layers
            layer_cumulative_attention = layer_attention.mean(dim=1).sum(dim=0).squeeze()
            cumulative_attention_scores += layer_cumulative_attention
        
        # Normalize attention scores
        cumulative_attention_scores /= cumulative_attention_scores.sum()
        
        # Determine if each token meets the threshold
        coverage = cumulative_attention_scores >= threshold
        sufficient_coverage = coverage.all().item()  # True if all tokens have sufficient attention
        
        # Prepare results
        coverage_details = {
            "cumulative_attention_scores": cumulative_attention_scores.tolist(),
            "sufficient_coverage": sufficient_coverage
        }
        
        return coverage_details

# Example usage
analyzer = PassageCoverageAnalyzer(model_name="meta-llama/Llama-3.2-1B")
input_text = "The llama wandered through the mountains."
coverage_results = analyzer.analyze_coverage(input_text)
print("Cumulative Attention Scores:", coverage_results["cumulative_attention_scores"])
print("Sufficient Coverage:", coverage_results["sufficient_coverage"])
