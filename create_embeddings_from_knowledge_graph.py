import os
import time
import logging
import networkx as nx
from typing import List
from openai import AzureOpenAI
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from GraphReasoning import save_embeddings
from tqdm import tqdm
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EmbeddingGenerator:
    """Handles text embedding generation using Azure OpenAI."""
    
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        )
        self.max_retries = 5
        self.backoff_factor = 1

    @sleep_and_retry
    @limits(calls=20, period=60)  # Adjust the rate limit as needed
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings with rate limiting and retries."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=[text],
                    model=os.getenv('AZURE_OPENAI_EMBEDDING_MODEL_NAME')
                )
                return response.data[0].embedding
            except Exception as e:
                wait_time = self.backoff_factor * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {wait_time}s: {str(e)}")
                time.sleep(wait_time)
                if attempt == self.max_retries - 1:
                    raise

def generate_node_embeddings(G, embedding_generator, max_workers=10, checkpoint_interval=100, checkpoint_dir="checkpoints"):
    embeddings = {}
    os.makedirs(checkpoint_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_node = {executor.submit(embedding_generator.generate_embedding, str(node)): node for node in G.nodes()}
        for i, future in enumerate(tqdm(as_completed(future_to_node), total=len(future_to_node))):
            node = future_to_node[future]
            try:
                embeddings[node] = future.result()
            except Exception as e:
                logger.error(f"Error generating embedding for node {node}: {e}")
            
            # Save checkpoint
            if (i + 1) % checkpoint_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(checkpoint_dir, f"embeddings_checkpoint_{timestamp}.pkl")
                save_embeddings(embeddings, checkpoint_path)
                logger.info(f"Checkpoint saved at {checkpoint_path}")
    return embeddings

def create_embeddings_from_knowledge_graph(
    data_dir: str = "knowledge_graph_paper_examples",
    graph_file_name: str = "knowledge_graph_graphML.graphml",
    checkpoint_interval: int = 100
):
    # Prepare directories
    output_dir = f"{data_dir}/embeddings"
    os.makedirs(output_dir, exist_ok=True)

    # Load the graph
    graph_path = f"{data_dir}/{graph_file_name}"
    G = nx.read_graphml(graph_path)
    logging.info(f"Graph loaded from {graph_path}")

    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator()

    # Generate embeddings
    checkpoint_dir = f"{output_dir}/checkpoints"
    node_embeddings = generate_node_embeddings(G, embedding_generator, checkpoint_interval=checkpoint_interval, checkpoint_dir=checkpoint_dir)

    # Save final embeddings
    embeddings_path = f"{output_dir}/node_embeddings.pkl"
    save_embeddings(node_embeddings, embeddings_path)

    print(f"Embeddings saved to {embeddings_path}")

if __name__ == "__main__":
    create_embeddings_from_knowledge_graph(
        data_dir="knowledge_graph_paper_examples",
        checkpoint_interval=100
    )