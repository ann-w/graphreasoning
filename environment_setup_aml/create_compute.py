import os 
from azure.ai.ml import MLClient
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from argparse import ArgumentParser

def get_ml_client():
    credential = DefaultAzureCredential()

    load_dotenv()

    subscription_id = os.getenv("SUBSCRIPTION_ID") 
    resource_group_name = os.getenv("RESOURCE_GROUP_NAME")
    workspace_name = os.getenv("AML_WORKSPACE_NAME")

    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )

    return ml_client


def main():
    parser = ArgumentParser(description = "Create compute resources.")
    parser.add_argument('--type', type=str, default='gpu', choices=['gpu', 'cpu'], help='Type of compute resource to create (gpu or cpu)')
    args = parser.parse_args()
    compute_type = args.type

    ml_client = get_ml_client()
    
    cpu_cluster_name = os.getenv("CPU_CLUSTER_NAME")
    gpu_cluster_name = os.getenv("GPU_CLUSTER_NAME")
    
    cluster = None

    if compute_type == "cpu":
        try:
            cluster = ml_client.compute.get(cpu_cluster_name)
            print(
                f"You already have a CPU cluster named {cpu_cluster_name}."
            )
            return

        except Exception:
            print("Creating a new CPU compute target...")
            new_compute_cluster = AmlCompute(
                    name=cpu_cluster_name,
                    type="amlcompute",
                    size="Standard_D16as_v4",
                    min_instances=0,
                    max_instances=2, # number of nodes
                    idle_time_before_scale_down=180,
                    tier="Dedicated",
                )  
    else:
        try:
            cluster = ml_client.compute.get(gpu_cluster_name)
            print(
                f"You already have a GPU cluster named {gpu_cluster_name}."
            )
            return 
        
        except Exception:
            print("Creating a new GPU compute target...")
            new_compute_cluster = AmlCompute(
                name=gpu_cluster_name,
                type="amlcompute",
                size="Standard_NC4as_T4_v3",
                min_instances=0,
                max_instances=1,
                idle_time_before_scale_down=180,
                )

    creation_op = ml_client.compute.begin_create_or_update(new_compute_cluster)
        
    while creation_op.status() != "Succeeded":
        print(f"Waiting for cluster creation... Status: {creation_op.status()}")
        creation_op.wait(timeout=5)

    cluster = creation_op.result()

    print(
        f"AMLCompute with name {cluster.name} is created, the compute size is {cluster.size}"
    )


if __name__ == "__main__":
    main()