import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random
import os
import urllib.request
import tarfile


def download_and_load_cora():
    """Download and load the Cora dataset"""
    print("Preparing Cora dataset...")

    # Download if not exists
    if not os.path.exists('cora'):
        os.makedirs('cora')

    if not os.path.exists('cora/cora.cites') or not os.path.exists('cora/cora.content'):
        print("Downloading Cora dataset...")
        url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
        urllib.request.urlretrieve(url, "cora.tgz")

        # Extract
        with tarfile.open("cora.tgz", "r:gz") as tar:
            tar.extractall()
        print("Dataset downloaded and extracted.")

    # Load citation data
    citations = pd.read_csv('cora/cora.cites', sep='\t', header=None, names=['citing', 'cited'])
    print(f"Citation edges: {len(citations)}")

    # Load content data
    content = pd.read_csv('cora/cora.content', sep='\t', header=None)
    paper_ids = content[0].values
    features = content.iloc[:, 1:-1].values
    labels = content.iloc[:, -1].values

    # Create graph
    G = nx.from_pandas_edgelist(citations, 'citing', 'cited', create_using=nx.DiGraph())

    # Create node attribute dictionary for category
    node_categories = {}
    for i, paper_id in enumerate(paper_ids):
        if paper_id in G.nodes():
            G.nodes[paper_id]['category'] = labels[i]
            node_categories[paper_id] = labels[i]

    print(f"Graph contains {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"Found categories for {len(node_categories)} papers")

    return G, node_categories


def visualize_network_sample(G, node_categories):
    """Create a visualization of the Cora citation network"""

    print("Creating citation network visualization...")

    # Find papers with categories and with at least one citation connection
    valid_nodes = []
    for node in G.nodes():
        if (node in node_categories and
                (G.in_degree(node) > 0 or G.out_degree(node) > 0)):
            valid_nodes.append(node)

    if len(valid_nodes) < 5:
        print("Error: Not enough connected nodes with categories")
        return

    # Select a connected component to ensure we have a connected graph
    seed_node = random.choice(valid_nodes)
    # Get nodes within 2 steps of the seed (limit to 20 nodes for clarity)
    subgraph_nodes = set([seed_node])
    neighbors = set()

    # Add outgoing connections (cited papers)
    neighbors.update([n for n in G.successors(seed_node) if n in node_categories])
    # Add incoming connections (citing papers)
    neighbors.update([n for n in G.predecessors(seed_node) if n in node_categories])

    # Add some of these neighbors to our subgraph
    sample_neighbors = list(neighbors)[:min(15, len(neighbors))]
    subgraph_nodes.update(sample_neighbors)

    # For each neighbor, add some of their connections too
    for neighbor in sample_neighbors:
        # Add a few connections from each neighbor
        next_neighbors = list(G.successors(neighbor)) + list(G.predecessors(neighbor))
        next_neighbors = [n for n in next_neighbors if n in node_categories]
        if next_neighbors:
            subgraph_nodes.update(random.sample(next_neighbors,
                                                min(2, len(next_neighbors))))

    # Create subgraph
    subgraph = G.subgraph(subgraph_nodes)

    # Set up colors by category
    unique_categories = sorted(set(node_categories[node] for node in subgraph_nodes))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}

    # Get node colors
    node_colors = [color_map[node_categories[node]] for node in subgraph.nodes()]

    # Create plot
    plt.figure(figsize=(10, 8))

    # Use spring layout with fixed seed for reproducibility
    pos = nx.spring_layout(subgraph, seed=42, k=0.3)

    # Draw nodes with categories
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=150)

    # Draw edges with arrows to show citation direction
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, arrows=True, arrowsize=15)

    # Add labels for a few key nodes
    # Select some nodes for labeling (not all, to avoid clutter)
    label_nodes = list(subgraph.nodes())[:min(7, len(subgraph.nodes()))]
    node_labels = {node: str(node) for node in label_nodes}
    nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=8)

    # Add legend
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=color_map[cat],
                                 markersize=10, label=cat) for cat in unique_categories]
    plt.legend(handles=legend_patches, loc='upper right')

    plt.title('Cora Citation Network Sample')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('citation_network_sample.png', dpi=300)
    plt.close()

    print(f"Created visualization with {len(subgraph.nodes())} papers and {len(subgraph.edges())} citations")
    print(f"Saved as citation_network_sample.png")
    return subgraph


# Execute the script
if __name__ == "__main__":
    # Download and load the Cora dataset
    G, node_categories = download_and_load_cora()

    # Create and save the visualization
    visualize_network_sample(G, node_categories)