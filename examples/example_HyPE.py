from models.HyPE import HyPE

def main():
    # Initialize HyPE with the default configuration
    hype = HyPE(config_path="config/config.yaml")
    
    # Example conversation history and retrieved documents
    conversation_history = [
        "User: Hi, I'm looking for a new laptop.",
        "Assistant: Sure, what are your requirements?",
        "User: I need something lightweight with at least 16GB RAM. Also, my SSN is 123-45-6789."
    ]
    
    retrieved_docs = [
        "Document 1: Recent reviews on laptops with 16GB RAM.",
        "Document 2: Best lightweight laptops in 2024.",
        "User's email: user@example.com"
    ]
    
    # Generate Hypothetical Ground Truths
    ground_truths = hype.generate_hypothetical_ground_truths(conversation_history, retrieved_docs)
    
    print("Generated Hypothetical Ground Truths:")
    for idx, gt in enumerate(ground_truths, 1):
        print(f"\nGround Truth {idx}:\n{gt}")

if __name__ == "__main__":
    main()
