# GraphReasoning Notebooks

This project explores graph-based reasoning techniques for scientific discovery, leveraging both analytical methods and large language models.

## Notebooks

1. **GraphReasoning - Graph Analysis.ipynb**  
   - Loads a BioGraph with network properties (nodes, edges).  
   - Analyzes embeddings, clusters, and community structures.  
   - Demonstrates various functions like:
     - `visualize_embeddings_2d_pretty_and_sample`  
     - `describe_communities_with_plots_complex` (requires correct `community_louvain` import)  
     - `find_path` and `find_best_fitting_node_list` for keyword-based queries.  

2. **GraphReasoning - Graph Reasoning with LLM - BioMixtral.ipynb**  
   - Loads and initializes a local large language model (“BioMixtral”).  
   - Integrates node embeddings, HF Hub downloads, and a specialized `generate_BioMixtral` function.  
   - Demonstrates how to run queries and handle results (paths, triplets, reasoning).  

