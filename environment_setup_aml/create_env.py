from azure.ai.ml import MLClient
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
import os
from azure.ai.ml.entities import Environment

credential = DefaultAzureCredential()

load_dotenv()
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group_name = os.getenv("RESOURCE_GROUP_NAME")
workspace_name = os.getenv("AML_WORKSPACE_NAME")
environment_name = os.getenv("ENVIRONMENT_NAME")

credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
)

conda_environment = Environment(
    name=environment_name,
    description="Environment based with PyTorch 2.2.1 and CUDA 12.1 support",
    conda_file="environment.yaml",
    image="pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel",
)

pipeline_job_env = ml_client.environments.create_or_update(conda_environment)

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
)
