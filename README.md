# GraphReasoning

This repository contains the graphreasoning library and scripts to create a knowledge graph.

## Create conda environment

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





## Creating a knowledge graph

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

## Reasoning over the knowledge graph

If you'd like to try out reasoning over a knowledge graph, you can run this notebook: `GraphReasoning/libs/GraphReasoning/Notebooks/GraphReasoning - Graph Reasoning with LLM - BioMixtral.ipynb`

<!-- 
## Setup AML Resources

You need the following resources in Azure:

- Azure Machine Learning -->

<!-- 
- Setup environment with terraform
had to enable storage access from all networks to create an environment
- cpu-cluster: Standard_D13_v2 (8 cores, 56 GB RAM, 400 GB disk) -> todo: change this in the terraform? 
- roles to assign to Storage Account. Copy principal and assign blob storage data reader and blob storage data owner and 
Storage Blob Data Contributor, 
- storage account > networking > allow from selected networks
- AML > networking > allow from selected networks -->

<!-- ### Create AML Environment 

- Go to `environment_setup_aml` folder
- Run in terminal:
    ```bash
    python create_env.py
    ``` -->