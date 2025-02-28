import os
import argparse
import networkx as nx
from datetime import datetime
from pyvis.network import Network
from azure_openai_client import initialize_clients, create_generate_function_wrapper
from GraphReasoning import load_embeddings, find_best_fitting_node_list_with_embedding
from utils import save_response_to_csv


def get_relevant_nodes(query_embedding, node_embeddings, top_k=5):
    """Retrieve the most relevant nodes for a query.
    
    Args:
        query_embedding: The embedding of the query
        node_embeddings: Dictionary of node embeddings
        top_k: Number of most relevant nodes to return
        
    Returns:
        List of most relevant node names and their similarities
    """
    # Use find_best_fitting_node_list_with_embedding instead of calculate_cosine_similarities
    node_similarities = find_best_fitting_node_list_with_embedding(
        query_embedding, node_embeddings, top_k=top_k
    )
    
    # Extract nodes and similarities from the result
    relevant_nodes = [node for node, _ in node_similarities]
    similarities = [sim for _, sim in node_similarities]
    
    return relevant_nodes, similarities


def find_and_visualize_path(G, source, target, data_dir="./", verbatim=False):
    """Find the shortest path between two nodes and visualize it.
    
    Args:
        G: The knowledge graph
        source: Source node name
        target: Target node name
        data_dir: Directory to save visualizations
        verbatim: Whether to print visualization path
        
    Returns:
        Tuple of (path, path_graph, shortest_path_length, fname, graph_GraphML)
    """
    try:
        # Find the shortest path between two nodes
        path = nx.shortest_path(G, source=source, target=target)
        shortest_path_length = nx.shortest_path_length(G, source=source, target=target)
        path_graph = G.subgraph(path)

        # Create safe versions of source and target for filenames
        safe_source = str(source).replace(" ", "_").replace("/", "_")[:20]
        safe_target = str(target).replace(" ", "_").replace("/", "_")[:20]

        # Create HTML visualization using pyvis
        try:
            nt = Network("500px", "1000px", notebook=False)
            nt.from_nx(path_graph)
            
            # Set title for the graph
            nt.set_options("""
            var options = {
                "nodes": {
                    "font": {
                        "size": 12
                    }
                },
                "edges": {
                    "font": {
                        "size": 12
                    }
                },
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
                    "stabilization": {
                        "enabled": true,
                        "iterations": 1000
                    }
                }
            }
            """)
            
            # Provide a title for the network
            fname = f"{data_dir}/shortest_path_{safe_source}_{safe_target}.html"
            
            # Use write_html instead of show as it's more reliable
            nt.write_html(fname, notebook=False)
            
            if verbatim:
                print(f"Visualization: {fname}")
                
        except Exception as e:
            # If visualization fails, just log it and continue
            print(f"Warning: Failed to create visualization for path {source} to {target}: {str(e)}")
            fname = None

        # Write graphML regardless of HTML visualization success
        graph_GraphML = f"{data_dir}/shortestpath_{safe_source}_{safe_target}.graphml"
        nx.write_graphml(path_graph, graph_GraphML)

        return path, path_graph, shortest_path_length, fname, graph_GraphML
    
    except nx.NetworkXNoPath:
        return None, None, float('inf'), None, None


def extract_paths(G, relevant_nodes, max_depth=2, data_dir="./"):
    """Extract paths between relevant nodes and visualize them.
    
    Args:
        G: The knowledge graph
        relevant_nodes: List of relevant node names
        max_depth: Maximum path length
        data_dir: Directory to save visualizations
        
    Returns:
        List of paths and their data
    """
    paths_data = []
    os.makedirs(data_dir, exist_ok=True)
    
    for i, source in enumerate(relevant_nodes):
        for target in relevant_nodes[i+1:]:  # Avoid duplicates by starting from i+1
            path, path_graph, path_length, fname, graph_GraphML = find_and_visualize_path(
                G, source, target, data_dir=data_dir
            )
            
            if path and path_length <= max_depth + 1:
                paths_data.append({
                    'path': path,
                    'path_graph': path_graph,
                    'path_length': path_length,
                    'visualization_file': fname,
                    'graphml_file': graph_GraphML
                })
    
    return paths_data


def create_prompt(query, G, relevant_nodes, similarities, paths_data):
    """Generate context-rich prompt from nodes and paths.
    
    Args:
        query: The user query text
        G: The knowledge graph
        relevant_nodes: List of relevant node names
        similarities: Similarity scores for each node
        paths_data: List of path data dictionaries
        
    Returns:
        Formatted prompt for LLM
    """
    # Format node descriptions with relevance scores
    node_descriptions = "\n".join([
        f"Node: {node}\nRelevance: {similarities[i]:.2f}\nDescription: {G.nodes[node].get('description', 'No description')}"
        for i, node in enumerate(relevant_nodes)
    ])

    # Format path descriptions
    path_descriptions = []
    for path_info in paths_data:
        path = path_info['path']
        path_length = path_info['path_length']
        
        path_text = " -> ".join([f"{n} ({G.nodes[n].get('description', 'No description')})" for n in path])
        path_descriptions.append(f"Path (length {path_length-1}): {path_text}")
    
    path_text = "\n\n".join(path_descriptions) if path_descriptions else "No paths found between relevant nodes."

    context = f"Relevant Nodes:\n{node_descriptions}\n\nRelevant Paths:\n{path_text}"

    system_prompt = "You are a helpful assistant that answers questions based on the provided knowledge graph information. Your answers should be grounded in the information provided."
    prompt = f"Answer the following query using only the information provided in the context. If the context doesn't contain enough information to answer the query, acknowledge that and provide the closest related information you can find.\n\nContext:\n{context}\n\nQuery: {query}"
    
    return system_prompt, prompt


def query_knowledge_graph(
    query_text,
    graph_path="data/output/graphs/knowledge_graph_graphML.graphml",
    embeddings_path="data/output/embeddings/node_embeddings.pkl",
    temperature=0.3,
    output_dir="data/output/qa_pairs",
    save_output_csv=True,
    top_k=5,
    max_path_depth=2
):
    """Query the knowledge graph with a free-form query.
    
    Args:
        query_text: User's query text
        graph_path: Path to the knowledge graph file
        embeddings_path: Path to the node embeddings file
        temperature: Temperature for text generation
        output_dir: Directory to save output files
        save_output_csv: Whether to save output to CSV
        top_k: Number of most relevant nodes to consider
        max_path_depth: Maximum path length to consider
        
    Returns:
        The LLM's response
    """
    # Set up output file path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a subdirectory for this specific query
    query_slug = query_text.replace(" ", "_")[:30]
    query_dir = f"{output_dir}/{timestamp}_{query_slug}"
    os.makedirs(query_dir, exist_ok=True)
    
    if save_output_csv:
        csv_path = f"{query_dir}/qa_log.csv"

    # Load graph and embeddings
    G = nx.read_graphml(graph_path)
    node_embeddings = load_embeddings(embeddings_path)

    # Initialize clients
    azure_openai_client, embedding_generator = initialize_clients()
    generate_func = create_generate_function_wrapper(azure_openai_client, temperature)
    
    # Get query embedding
    query_embedding = embedding_generator.generate_embedding(query_text)
    
    # Find relevant nodes
    relevant_nodes, similarities = get_relevant_nodes(
        query_embedding, 
        node_embeddings, 
        top_k=top_k
    )
    
    # Create visualizations directory
    vis_dir = f"{query_dir}/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract paths between relevant nodes and visualize them
    paths_data = extract_paths(
        G, 
        relevant_nodes, 
        max_depth=max_path_depth,
        data_dir=vis_dir
    )
    
    # Create prompt for LLM
    system_prompt, prompt = create_prompt(query_text, G, relevant_nodes, similarities, paths_data)
    
    # Get response from LLM
    response = generate_func(system_prompt=system_prompt, prompt=prompt, temperature=temperature)
    
    # Save the prompt and response to a text file
    with open(f"{query_dir}/prompt_and_response.txt", "w") as f:
        f.write(f"System Prompt:\n{system_prompt}\n\nPrompt:\n{prompt}\n\nResponse:\n{response}")
    
    print("---------- QUERY RESPONSE ----------")
    print(response)
    print(f"\nVisualizations and output saved to: {query_dir}")
    
    if save_output_csv:
        save_response_to_csv(csv_path, query_text, response, system_prompt, prompt)
        
    return response


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query a knowledge graph with a natural language question."
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="The query text to answer using the knowledge graph"
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
        default="data/output/graphs/knowledge_graph_graphML.graphml",
        help="Path to the knowledge graph file"
    )
    
    parser.add_argument(
        "--embeddings-path", 
        type=str, 
        default="data/output/embeddings/node_embeddings.pkl",
        help="Path to the node embeddings file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/output/qa_pairs",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--no-save", 
        action="store_false", 
        dest="save_output_csv",
        help="Don't save output to CSV"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5,
        help="Number of most relevant nodes to consider (default: 5)"
    )
    
    parser.add_argument(
        "--max-path-depth", 
        type=int, 
        default=2,
        help="Maximum path length between nodes (default: 2)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    query_knowledge_graph(
        query_text=args.query,
        graph_path=args.graph_path,
        embeddings_path=args.embeddings_path,
        temperature=args.temperature,
        output_dir=args.output_dir,
        save_output_csv=args.save_output_csv,
        top_k=args.top_k,
        max_path_depth=args.max_path_depth
    )