import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import urllib.request
import tarfile


# Download Cora dataset if not already present
def download_cora():
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


download_cora()

# Load citation data
citations = pd.read_csv('cora/cora.cites', sep='\t', header=None, names=['citing', 'cited'])
print(f"Citation edges: {len(citations)}")

# Load content data
content = pd.read_csv('cora/cora.content', sep='\t', header=None)
paper_ids = content[0].values
features = content.iloc[:, 1:-1].values
labels = content.iloc[:, -1].values

# Create mapping from paper IDs to indices
id_to_idx = {paper_id: i for i, paper_id in enumerate(paper_ids)}

# Create graph
G = nx.from_pandas_edgelist(citations, 'citing', 'cited', create_using=nx.DiGraph())

# Map categories to integers for easier processing
unique_labels = sorted(set(labels))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
idx_to_label = {i: label for label, i in label_to_idx.items()}
y = np.array([label_to_idx[label] for label in labels])

print(f"Papers: {len(paper_ids)}")
print(f"Features per paper: {features.shape[1]}")
print(f"Categories: {unique_labels}")

# Create node attribute dictionary for category
node_categories = {}
for i, paper_id in enumerate(paper_ids):
    if paper_id in G.nodes():
        G.nodes[paper_id]['category'] = labels[i]
        node_categories[paper_id] = labels[i]

# Create feature matrix with papers in the same order as they appear in the graph
feature_dict = {paper_id: features[i] for i, paper_id in enumerate(paper_ids)}

# If this file is run directly
if __name__ == "__main__":
    print("Cora dataset loaded successfully")