import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# Import needed modules - these imports will work with renamed files
from cora_preprocessing import G, node_categories, label_to_idx
from cora_embeddings import node_embeddings, visualize_embeddings
from hybrid_mln import HybridMarkovLogic

print("Starting MLN verification process...")

# Create MLN instance
mln = HybridMarkovLogic(G, node_embeddings, node_categories)

# Add rules with weights
mln.add_rule("citation_category", weight=2.0)
mln.add_rule("co_citation_category", weight=1.5)
mln.add_rule("embedding_similarity", weight=3.0)

# Verify embeddings
verified_embeddings = mln.verify_embeddings(iterations=10)  # Reduced iterations for speed

# Evaluate verification
results = mln.evaluate_verification()
print(f"Original rule satisfaction: {results['original_avg_satisfaction']:.4f}")
print(f"Verified rule satisfaction: {results['verified_avg_satisfaction']:.4f}")
print(f"Improvement: {results['improvement']:.4f}")

# Sample a subset of nodes for visualization
sample_size = min(500, len(G.nodes()))
sample_nodes = random.sample([n for n in G.nodes() if n in node_categories], sample_size)

# Visualize verified embeddings
visualize_embeddings(
    {node: verified_embeddings.get(node, node_embeddings[node]) for node in sample_nodes},
    node_categories, sample_nodes,
    title="Verified Node Embeddings"
)


# Compare original vs verified embeddings in node classification task
def evaluate_embeddings_for_classification(embeddings, labels):
    # Create feature matrix and labels array
    nodes = [node for node in embeddings.keys() if node in labels]
    if not nodes:
        print("No labeled nodes found in the embeddings!")
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    X = np.array([embeddings[node] for node in nodes])
    y = np.array([label_to_idx[labels[node]] for node in nodes])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# Evaluate both embedding types on common nodes
common_nodes = [node for node in sample_nodes if node in verified_embeddings]
orig_sample = {node: node_embeddings[node] for node in common_nodes}
verified_sample = {node: verified_embeddings[node] for node in common_nodes}

print("\nEvaluating classification performance...")
original_metrics = evaluate_embeddings_for_classification(orig_sample, node_categories)
verified_metrics = evaluate_embeddings_for_classification(verified_sample, node_categories)

print("\nOriginal Embeddings Classification Metrics:")
for metric, value in original_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nVerified Embeddings Classification Metrics:")
for metric, value in verified_metrics.items():
    print(f"{metric}: {value:.4f}")

# Create comparison plots
metrics = ['accuracy', 'precision', 'recall', 'f1']
original_values = [original_metrics[m] for m in metrics]
verified_values = [verified_metrics[m] for m in metrics]

# Bar chart comparing metrics
plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width / 2, original_values, width, label='Original Embeddings')
plt.bar(x + width / 2, verified_values, width, label='Verified Embeddings')

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Classification Performance: Original vs. Verified Embeddings')
plt.xticks(x, metrics)
plt.legend()
plt.tight_layout()
plt.savefig('embedding_comparison_metrics.png', dpi=300)
plt.close()

# Visualize rule satisfaction improvement
plt.figure(figsize=(8, 5))
satisfaction = [results['original_avg_satisfaction'], results['verified_avg_satisfaction']]
plt.bar(['Original Embeddings', 'Verified Embeddings'], satisfaction, color=['#1f77b4', '#ff7f0e'])
plt.ylabel('Rule Satisfaction')
plt.title('MLN Rule Satisfaction Before and After Verification')
plt.ylim(0, 1.0)
for i, v in enumerate(satisfaction):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.tight_layout()
plt.savefig('rule_satisfaction.png', dpi=300)
plt.close()


# Create network visualization with categories
def visualize_network_sample():
    # Sample a subset of nodes for visualization
    labeled_nodes = [node for node in G.nodes() if node in node_categories]
    if len(labeled_nodes) < 3:
        print("Not enough labeled nodes for visualization")
        return

    seeds = random.sample(labeled_nodes, 3)
    subgraph_nodes = set(seeds)

    # Add neighbors
    for seed in seeds:
        successors = [n for n in G.successors(seed) if n in node_categories]
        predecessors = [n for n in G.predecessors(seed) if n in node_categories]
        subgraph_nodes.update(successors[:min(5, len(successors))])
        subgraph_nodes.update(predecessors[:min(5, len(predecessors))])

    # Create subgraph
    subgraph = G.subgraph(subgraph_nodes)

    # Set up colors by category
    unique_categories = sorted(set(node_categories.values()))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}

    node_colors = [color_map[node_categories[node]] if node in node_categories else '#cccccc' for node in
                   subgraph.nodes()]

    # Create plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=100)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, arrows=True)

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


visualize_network_sample()

# Write MLN Rules to file for presentation
with open('mln_rules.txt', 'w') as f:
    f.write("# Hybrid Markov Logic Rules for Embedding Verification\n\n")
    f.write("## Citation Category Agreement Rule\n")
    f.write("2.0 Category(p1, c) ^ Cites(p1, p2) => Category(p2, c)\n\n")
    f.write("## Co-Citation Agreement Rule\n")
    f.write("1.5 Cites(p3, p1) ^ Cites(p3, p2) => SameCategory(p1, p2)\n\n")
    f.write("## Embedding Similarity Rule\n")
    f.write("3.0 SimilarEmbedding(p1, p2) => SameCategory(p1, p2)\n")

print("Analysis complete. Generated visualizations for your presentation.")