# GraphReasoning on AML

## Create conda environment

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if you haven't already.

2. Create a new Conda environment:

    ```bash
    conda create --name graphreasoning-aml python=3.10
    ```

3. Activate the Conda environment:

    ```bash
    conda activate graphreasoning-aml
    ```

4. Install the required dependencies:

    ```bash
    pip install --no-cache-dir -r requirements.txt
    ```


## Setup AML Resources

You need the following resources in Azure:

- Azure Machine Learning

<!-- 
- Setup environment with terraform
had to enable storage access from all networks to create an environment
- cpu-cluster: Standard_D13_v2 (8 cores, 56 GB RAM, 400 GB disk) -> todo: change this in the terraform? 
- roles to assign to Storage Account. Copy principal and assign blob storage data reader and blob storage data owner and 
Storage Blob Data Contributor, 
- storage account > networking > allow from selected networks
- AML > networking > allow from selected networks -->

### Create AML Environment 

- Go to `environment_setup_aml` folder
- Run in terminal:
    ```bash
    python create_env.py
    ```