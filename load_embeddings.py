import pickle

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def inspect_embeddings(embeddings):
    print(f"Number of nodes: {len(embeddings)}")
    for node, embedding in list(embeddings.items())[:5]:  # Print the first 5 embeddings
        print(f"Node: {node}, Embedding: {embedding[:10]}...")  # Print the first 10 dimensions of the embedding

if __name__ == "__main__":
    embeddings_path = "./knowledge_graph_paper_examples/embeddings/node_embeddings.pkl"
    embeddings = load_embeddings(embeddings_path)
    inspect_embeddings(embeddings)