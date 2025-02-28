import os
import csv
import networkx as nx
from datetime import datetime
from azure_openai_client import AzureOpenAIClient
from create_embeddings_from_knowledge_graph import EmbeddingGenerator
from GraphReasoning import load_embeddings, find_path_and_reason


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


def save_response_to_csv(csv_path, question_text, response):
    """Save the question and response to a CSV file."""
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["question", "answer"])
        writer.writerow([question_text, response])


def query_graph_with_openai(
    keyword_1="cement",
    keyword_2="health",
    graph_path="knowledge_graph_paper_examples/knowledge_graph_graphML.graphml",
    embeddings_path="knowledge_graph_paper_examples/embeddings/node_embeddings.pkl",
    temperature=0.3,
    output_dir="knowledge_graph_paper_examples/qa_pairs",
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

    response, (best_node_1, _, best_node_2, _), path, path_graph, shortest_path_length, fname, graph_GraphML = find_path_and_reason(
        G=G,
        node_embeddings=node_embeddings,
        generate=lambda system_prompt, prompt: generate_response(azure_openai_client, system_prompt, prompt, temperature),
        keyword_1=keyword_1,
        keyword_2=keyword_2,
        output_dir=output_dir,
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


if __name__ == "__main__":
    query_graph_with_openai(keyword_1="cement", keyword_2="health", temperature=0.3)