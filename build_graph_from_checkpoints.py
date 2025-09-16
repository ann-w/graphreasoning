#!/usr/bin/env python3
import json
import glob
import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx

def load_checkpoints(checkpoint_glob):
    files = sorted(glob.glob(checkpoint_glob))
    all_triplets = []
    for f in files:
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
                # data may be nested lists (list per chunk); flatten
                for item in data:
                    if isinstance(item, list):
                        all_triplets.extend(item)
                    else:
                        all_triplets.append(item)
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")
    return all_triplets

def build_graph_from_triplets(triplets, graph_out_path, data_dir="./data/output/graphs"):
    # Normalize: ensure keys node_1,node_2,edge,chunk_id exist
    df = pd.DataFrame(triplets)
    for c in ["node_1","node_2","edge","chunk_id"]:
        if c not in df.columns:
            df[c] = np.nan
    # cleanup
    df = df.replace("", np.nan).dropna(subset=["node_1","node_2"])
    df["node_1"] = df["node_1"].astype(str).str.strip()
    df["node_2"] = df["node_2"].astype(str).str.strip()
    # aggregate edges similarly to make_graph_from_text
    dfg = (df.groupby(["node_1","node_2"])
             .agg({"chunk_id":",".join, "edge":",".join})
             .reset_index())
    dfg["count"] = 1
    # build graph
    G = nx.Graph()
    nodes = pd.concat([dfg["node_1"], dfg["node_2"]], axis=0).unique()
    for n in nodes:
        G.add_node(str(n))
    for _, row in dfg.iterrows():
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            weight=float(row["count"]),
            chunk_id=row["chunk_id"]
        )
    os.makedirs(os.path.dirname(graph_out_path), exist_ok=True)
    nx.write_graphml(G, graph_out_path)
    # also save nodes/edges CSVs
    nodes_df = pd.DataFrame({"nodes": list(G.nodes())})
    nodes_df.to_csv(os.path.join(data_dir, "knowledge_graph_nodes.csv"), index=False)
    dfg.to_csv(os.path.join(data_dir, "knowledge_graph_edges.csv"), index=False)
    print(f"Graph saved to {graph_out_path}  (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()})")
    return G

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", default="data/output/graphs/checkpoints/triplets_checkpoint_*.json")
    p.add_argument("--out", default="data/output/graphs/knowledge_graph_graphML.graphml")
    args = p.parse_args()

    triplets = load_checkpoints(args.checkpoints)
    if not triplets:
        raise SystemExit("No triplets found. Check checkpoint files path.")
    build_graph_from_triplets(triplets, args.out)