import os
import argparse
import networkx as nx
from datetime import datetime
from azure_openai_client import AzureOpenAIClient
from create_embeddings_from_knowledge_graph import EmbeddingGenerator
from GraphReasoning import load_embeddings, find_path_and_reason
from utils import save_response_to_csv

def initialize_clients():
    """Initialize Azure OpenAI client and embedding generator."""
    azure_openai_client = AzureOpenAIClient()
    embedding_generator = EmbeddingGenerator()
    return azure_openai_client, embedding_generator


def generate_response(azure_openai_client, system_prompt, prompt, temperature):
    """Generate response using Azure OpenAI."""
    full_prompt = f"{system_prompt}\n{prompt}"
    response = azure_openai_client.generate_completion(prompt=full_prompt, temperature=temperature)
    return response


def create_generate_function(client, temperature_value):
    """Create a generate function for OpenAI API calls.
    
    Args:
        client: The Azure OpenAI client
        temperature_value: Temperature value for generation
        
    Returns:
        A function that can be passed to find_path_and_reason
    """
    def generate_func(system_prompt, prompt, max_tokens=4096, temperature=None):
        temp = temperature if temperature is not None else temperature_value
        return generate_response(
            azure_openai_client=client,
            system_prompt=system_prompt, 
            prompt=prompt, 
            temperature=temp
        )
    return generate_func


def query_graph_with_openai(
    keyword_1="cement",
    keyword_2="health",
    graph_path="data/output/graphs/knowledge_graph_graphML.graphml",
    embeddings_path="data/output/embeddings/node_embeddings.pkl",
    temperature=0.3,
    output_dir="data/output/qa_pairs",
    save_output_csv=True,
):
    """Query the graph and generate a response using Azure OpenAI."""
    
    if save_output_csv:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = f"{output_dir}/{timestamp}_qa_log.csv"

    question_text = f"Develop a new research idea around {keyword_1} and {keyword_2}."

    G = nx.read_graphml(graph_path)
    node_embeddings = load_embeddings(embeddings_path)

    azure_openai_client, embedding_generator = initialize_clients()

    generate_func = create_generate_function(azure_openai_client, temperature)


    response, (best_node_1, _, best_node_2, _), path, path_graph, shortest_path_length, fname, graph_GraphML = find_path_and_reason(
        G=G,
        node_embeddings=node_embeddings,
        generate=generate_func,
        keyword_1=keyword_1,
        keyword_2=keyword_2,
        data_dir=output_dir,
        instruction=f"Develop a new research idea around {keyword_1} and {keyword_2}.",
        include_keywords_as_nodes=True,
        visualize_paths_as_graph=True,
        display_graph=False,
        temperature=temperature,
        inst_prepend="### ",
        prepend="You are given a set of information from a graph describing relationships among materials.\n\n",
        embedding_generator=embedding_generator
    )

    print("---------- QUERY RESPONSE ----------")
    print(response)

    if save_output_csv:
        save_response_to_csv(csv_path, question_text, response)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reason over a knowledge graph with two keywords."
    )
    
    parser.add_argument(
        "--keyword1", 
        type=str, 
        default="cement",
        help="First keyword to reason about (default: cement)"
    )
    
    parser.add_argument(
        "--keyword2", 
        type=str, 
        default="health",
        help="Second keyword to reason about (default: health)"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.3,
        help="Temperature for text generation (default: 0.3)"
    )
    
    parser.add_argument(
        "--graph-path", 
        type=str, 
        default="knowledge_graph_paper_examples/knowledge_graph_graphML.graphml",
        help="Path to the knowledge graph file"
    )
    
    parser.add_argument(
        "--embeddings-path", 
        type=str, 
        default="knowledge_graph_paper_examples/embeddings/node_embeddings.pkl",
        help="Path to the node embeddings file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="knowledge_graph_paper_examples/qa_pairs",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--no-save", 
        action="store_false", 
        dest="save_output_csv",
        help="Don't save output to CSV"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    query_graph_with_openai(
        keyword_1=args.keyword1,
        keyword_2=args.keyword2,
        graph_path=args.graph_path,
        embeddings_path=args.embeddings_path,
        temperature=args.temperature,
        output_dir=args.output_dir,
        save_output_csv=args.save_output_csv
    )