import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import random
import networkx as nx  # Add this import

# This ensures it can be imported without errors
try:
    from cora_preprocessing import G, node_categories
except ImportError:
    # If running directly, we need to execute preprocessing first
    exec(open("cora_preprocessing.py").read())
    from cora_preprocessing import G, node_categories


# Generate simplified embeddings
def generate_spectral_embeddings(G, dimensions=64):
    print("Generating spectral embeddings...")
    # Use spectral embedding instead of Node2Vec (for simplicity and reliability)
    from sklearn.decomposition import TruncatedSVD

    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).astype(float)

    # Apply SVD to get embeddings
    svd = TruncatedSVD(n_components=dimensions, random_state=42)
    embeddings_matrix = svd.fit_transform(adj_matrix)

    # Convert to dictionary
    embeddings = {}
    for i, node in enumerate(G.nodes()):
        embeddings[node] = embeddings_matrix[i]

    print(f"Generated {len(embeddings)} embeddings of dimension {dimensions}")
    return embeddings


# Generate embeddings for our graph
node_embeddings = generate_spectral_embeddings(G)


# Calculate embedding similarity between nodes
def calculate_similarity_matrix(embeddings, nodes):
    # Create embedding matrix where each row is the embedding of a node
    node_list = list(nodes)
    embedding_matrix = np.array([embeddings[node] for node in node_list])

    # Calculate cosine similarity between all pairs of embeddings
    similarity_matrix = cosine_similarity(embedding_matrix)

    # Create a dictionary to store similarities
    similarity_dict = {}
    for i, node1 in enumerate(node_list):
        for j, node2 in enumerate(node_list):
            if i != j:  # Skip self-similarities
                similarity_dict[(node1, node2)] = similarity_matrix[i, j]

    return similarity_dict, similarity_matrix, node_list


# Calculate similarity for a sample of nodes in the graph
# Using a sample to make computation faster
sample_nodes = list(G.nodes())[:1000]
node_similarities, sim_matrix, node_order = calculate_similarity_matrix(node_embeddings, sample_nodes)


# Visualize embeddings
def visualize_embeddings(embeddings, labels, node_list, title="Node Embeddings Visualization"):
    # Create embedding matrix
    embedding_matrix = np.array([embeddings[node] for node in node_list])

    # Label array in the same order as embedding_matrix
    label_array = np.array([labels.get(node, "Unknown") for node in node_list])

    # Apply t-SNE for dimensionality reduction
    print(f"Applying t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embedding_matrix)

    # Plot
    plt.figure(figsize=(12, 10))

    # Get unique categories for coloring
    unique_categories = sorted(set(label_array))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}

    # Plot each point
    for i, (x, y) in enumerate(embeddings_2d):
        category = label_array[i]
        plt.scatter(x, y, color=color_map[category], label=category)

    # Create legend without duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(title)
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

    return embeddings_2d


# If this file is run directly, visualize embeddings
if __name__ == "__main__":
    # Sample a subset for visualization (for speed)
    sample_size = min(500, len(G.nodes()))
    visualization_nodes = random.sample([n for n in G.nodes() if n in node_categories], sample_size)

    # Visualize original embeddings
    embeddings_2d = visualize_embeddings(
        {node: node_embeddings[node] for node in visualization_nodes},
        node_categories,
        visualization_nodes,
        title="Original Node Embeddings"
    )