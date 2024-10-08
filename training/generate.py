import torch

def generate_text(model, tokenizer, prompt, max_length=50):
    """
    Generates text using the provided model and tokenizer.
    :param model: The language model
    :param tokenizer: The tokenizer
    :param prompt: The input prompt
    :param max_length: Maximum length of the generated text
    :return: Generated text
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode the output and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
