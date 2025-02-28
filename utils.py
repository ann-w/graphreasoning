import os
import csv
from GraphReasoning import load_embeddings
from datetime import datetime

def save_response_to_csv(csv_path, question, response, system_prompt=None, prompt=None):
    """Save the QA pair and prompts to a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        question: The question/query text
        response: The LLM's response
        system_prompt: The system prompt (optional)
        prompt: The full prompt sent to the LLM (optional)
    """
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'question', 'response', 'system_prompt', 'full_prompt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow({
            'timestamp': timestamp,
            'question': question,
            'response': response,
            'system_prompt': system_prompt if system_prompt else "",
            'full_prompt': prompt if prompt else ""
        })


def inspect_embeddings(embeddings_path = "./data/output/embeddings/node_embeddings.pkl"):
    embeddings = load_embeddings(embeddings_path)
    inspect_embeddings(embeddings)
    print(f"Number of nodes: {len(embeddings)}")
    for node, embedding in list(embeddings.items())[
        :10
    ]:  # Print the [:index] embeddings
        print(
            f"Node: {node}, Embedding: {embedding[:10]}..."
        )  # Print the first 10 dimensions of the embedding
