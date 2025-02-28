# GraphReasoning

This repository contains the graphreasoning library and scripts to create a knowledge graph and is forked from [here](https://github.com/lamm-mit/GraphReasoning). 

Changes made:
- Add parallel API calls to speed up extraction of nodes and relationships
- Fix package versions
- Create script to create knowledge graph given a directory with pdf files

## Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if you haven't already.

2. Create a new Conda environment:

    ```bash
    conda create --name graphreasoning python=3.10
    ```

3. Activate the Conda environment:

    ```bash
    conda activate graphreasoning
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

> **Note:** If you are having issues with installing `python-cpp-llama` on ubuntu CPU, try the following command instead:

```bash
CMAKE_ARGS="-DLLAVA_BUILD=OFF" pip install -U llama-cpp-python
```

5. To run the notebooks you need to install ipykernel:
    ```
    pip install ipykernel
    python3 -m ipykernel install --user --name graphreasoning --display-name "Python3 (graphreasoning)"
    ```

## Step 1: Creating a knowledge graph

The folder `paper_examples` contains a set of sample papers from which we can create a graph. 
We will use Azure OpenAI to create this graph.

1. Copy `.env.template` as `.env` and fill in below values after you have deployed your openAI model on Azure:

    ```bash
    AZURE_OPENAI_API_KEY="<your-azure-openai-api-key>"
    AZURE_OPENAI_API_VERSION="<your-azure-openai-api-version>"
    AZURE_OPENAI_ENDPOINT="<your-azure-openai-endpoint>"
    AZURE_OPENAI_MODEL_NAME="<your-azure-openai-model-name>"
    ```

2. Create the graph using:

    ```
    python create_graph_from_papers.py
    ```

    or if you'd like to save the logs to a textfile, you can run:
    ```
    chmod +x ./run_create_graph_from_papers.sh
    ./run_create_graph_from_papers.sh
    ```

## Step 2: Embed the nodes of the knowledge graph

1. Fill in the following values in the `.env` file.

    ```
    AZURE_OPENAI_EMBEDDING_API_KEY="<your-azure-openai-embedding-api-key>"
    AZURE_OPENAI_EMBEDDING_API_VERSION="<your-azure-openai-embedding-api-version>"
    AZURE_OPENAI_EMBEDDING_ENDPOINT="<your-azure-openai-embedding-endpoint>"
    AZURE_OPENAI_EMBEDDING_MODEL_NAME="<your-azure-openai-embedding-model-name>"
    ```

2. To embed the nodes of the knowledge graph, run:

    ```bash
    python create_embeddings_from_knowledge_graph.py
    ```

## Step 3: Reason over the knowledge graph

The main reasoning functionality in GraphReasoning is centered around the `find_path_and_reason` function in `graph_analysis.py`, which requires explicit specification of two keywords to find connections between them in the knowledge graph.

It works as follows:

1. **Node Matching**: It first finds the best-fitting nodes in the graph that match your keywords by calculating semantic similarity between your keywords and the node embeddings.

2. **Path Identification**: Once the corresponding nodes are identified, the function finds the shortest path between these nodes in the knowledge graph.

3. **Subgraph Extraction**: It extracts a subgraph containing the path and related relationships, which provides the context for reasoning.

Example usage:

To reason with the keywords 'cement' and 'environment', run:

```bash
python reason_with_knowledge_graph.py --keyword1 "cement" --keyword2 "environment"
```