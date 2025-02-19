import os
from pathlib import Path
from typing import List, Callable, Optional, Any

from dotenv import load_dotenv
import logging
import networkx as nx
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from GraphReasoning.graph_generation import (
    make_graph_from_text,
    df2Graph_parallel,
    documents2Dataframe,
    df2Graph,
    graph2Df,
    graphPrompt
)
from azure_openai_client import AzureOpenAIClient

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_papers(directory: str, num_docs: Optional[int] = None) -> List[Any]:
    """
    Load PDF documents from the specified directory.

    Args:
        directory (str): The directory containing the PDF documents.
        num_docs (Optional[int]): Number of documents to load. If None, load all.

    Returns:
        List[Any]: A list of loaded documents.
    """
    loader = DirectoryLoader(directory, loader_cls=PyPDFLoader)
    documents = loader.load()
    if num_docs:
        documents = documents[:num_docs]
    return documents

def combine_documents_text(documents: List[Any]) -> str:
    """
    Combine text from a list of documents into a single string.
    
    Args:
        documents (List[Any]): The list of documents (each should have a 'page_content'
                               attribute or similar text property).
    
    Returns:
        str: A single string with all document text.
    """
    text_blocks = []
    for doc in documents:
        # Assuming each document has a property 'page_content'
        # Otherwise, fallback to the string representation of the document.
        text = getattr(doc, "page_content", str(doc))
        text_blocks.append(text)
    return "\n".join(text_blocks)

def generate_graph_from_papers(directory: str,
                               generate: Callable[[str, str], str],
                               data_dir: str = './knowledge_graph_paper_examples/',
                               num_docs: Optional[int] = None) -> nx.Graph:
    """
    Generate a knowledge graph from research papers by processing their text and
    saving intermediate results (e.g. CSV files for chunks and graph data).
    
    The function uses make_graph_from_text which splits the text, applies the LLM,
    saves intermediate CSVs (chunks/graph) and returns the final graph along with
    visualization outputs.

    Args:
        directory (str): Directory containing PDF documents.
        generate (Callable[[str, str], str]): Function to call the LLM.
        data_dir (str): Directory to save intermediate files and final graph.
        num_docs (Optional[int]): Number of documents to load.

    Returns:
        nx.Graph: The generated knowledge graph.
    """
    # Load and combine document texts.
    documents = load_papers(directory, num_docs=num_docs)
    all_text = combine_documents_text(documents)
    
    # Call make_graph_from_text to process text.
    # Parameters: 
    #   include_contextual_proximity=True to add extra edges (if desired),
    #   graph_root as the saved file prefix,
    #   chunk_size, chunk_overlap and other parameters as needed.

    # Default params:                           
    # chunk_size=2500,chunk_overlap=0,

    html, graphml, graph, net, pdf = make_graph_from_text(
        all_text,
        generate,
        include_contextual_proximity=True,  # investigate
        graph_root='knowledge_graph',
        chunk_size=2500,
        chunk_overlap=0,
        repeat_refine=0,
        verbatim=False,
        data_dir=data_dir,
        save_PDF=False,
        save_HTML=True
    )
    return graph

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a knowledge graph from research papers.")
    parser.add_argument("--num_docs", type=int, default=None, help="Number of documents to use from the folder.")
    args = parser.parse_args()

    directory = './paper_examples/'
    data_dir = './knowledge_graph_paper_examples/'

    # Initialize the AzureOpenAIClient for making API calls.
    azure_openai_client = AzureOpenAIClient()
    
    def generate(system_prompt: str, prompt: str) -> str:
        """
        Generate a response using AzureOpenAIClient.
        
        Constructs a list of messages and calls generate_response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return azure_openai_client.generate_response(messages)
    
    # Generate graph (which will also save intermediate results via make_graph_from_text).
    G = generate_graph_from_papers(directory, generate, data_dir=data_dir, num_docs=args.num_docs)
    print(f"Graph generated and saved to {data_dir}")

if __name__ == "__main__":
    main()