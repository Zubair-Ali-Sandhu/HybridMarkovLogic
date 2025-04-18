import numpy as np
from collections import defaultdict
import random


class HybridMarkovLogic:
    def __init__(self, graph, embeddings, categories):
        self.graph = graph
        self.embeddings = embeddings
        self.categories = categories
        self.rules = []
        self.weights = []
        self.verified_embeddings = None

        # Calculate initial similarity matrix
        self.embedding_sim = self._calculate_similarity()

    def _calculate_similarity(self):
        """Calculate pairwise cosine similarity between node embeddings"""
        sim_dict = {}
        # Use a sample of nodes for efficiency
        nodes = list(self.graph.nodes())[:1000]  # Use first 1000 nodes for efficiency
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j and node1 in self.embeddings and node2 in self.embeddings:
                    e1 = self.embeddings[node1]
                    e2 = self.embeddings[node2]
                    # Cosine similarity
                    num = np.dot(e1, e2)
                    denom = np.linalg.norm(e1) * np.linalg.norm(e2)
                    sim = num / denom if denom > 0 else 0
                    sim_dict[(node1, node2)] = sim
        return sim_dict

    def add_rule(self, rule_type, weight=1.0):
        """Add a logical rule to the MLN"""
        self.rules.append(rule_type)
        self.weights.append(weight)
        print(f"Added rule: {rule_type} with weight {weight}")

    def _citation_category_agreement(self, node_pair):
        """Rule: If A cites B, they likely have the same category"""
        node1, node2 = node_pair
        if self.graph.has_edge(node1, node2):
            if node1 in self.categories and node2 in self.categories:
                return 1.0 if self.categories[node1] == self.categories[node2] else 0.0
        return 0.5  # Neutral if no edge

    def _co_citation_category_agreement(self, node_pair):
        """Rule: If A and B are both cited by C, they likely have the same category"""
        node1, node2 = node_pair
        citations1 = set(self.graph.predecessors(node1)) if node1 in self.graph else set()
        citations2 = set(self.graph.predecessors(node2)) if node2 in self.graph else set()
        common_citations = citations1.intersection(citations2)

        if common_citations:
            if node1 in self.categories and node2 in self.categories:
                return 1.0 if self.categories[node1] == self.categories[node2] else 0.0
        return 0.5  # Neutral if no common citations

    def _embedding_similarity_implies_same_category(self, node_pair):
        """Rule: High embedding similarity implies same category"""
        node1, node2 = node_pair
        sim = self.embedding_sim.get((node1, node2), 0.0)

        if node1 in self.categories and node2 in self.categories:
            same_category = self.categories[node1] == self.categories[node2]
            # Satisfaction depends on both similarity and category match
            if sim > 0.8 and same_category:
                return 1.0  # High similarity + same category = satisfied
            elif sim > 0.8 and not same_category:
                return 0.0  # High similarity + different category = violated
            elif sim < 0.2 and not same_category:
                return 1.0  # Low similarity + different category = satisfied
            elif sim < 0.2 and same_category:
                return 0.0  # Low similarity + same category = violated

        # Default to neutral
        return 0.5

    def evaluate_rules(self, node_pair):
        """Evaluate all rules for a given node pair"""
        rule_satisfaction = []

        for rule in self.rules:
            if rule == "citation_category":
                satisfaction = self._citation_category_agreement(node_pair)
            elif rule == "co_citation_category":
                satisfaction = self._co_citation_category_agreement(node_pair)
            elif rule == "embedding_similarity":
                satisfaction = self._embedding_similarity_implies_same_category(node_pair)
            else:
                satisfaction = 0.5  # Unknown rule

            rule_satisfaction.append(satisfaction)

        return rule_satisfaction

    def verify_embeddings(self, iterations=10):
        """Adjust embeddings to better satisfy MLN rules"""
        print(f"Starting embedding verification with {iterations} iterations")

        # Copy original embeddings
        self.verified_embeddings = {node: np.copy(emb) for node, emb in self.embeddings.items()}

        # Adjustment rate
        learn_rate = 0.01

        for iteration in range(iterations):
            adjustments = defaultdict(lambda: np.zeros_like(next(iter(self.verified_embeddings.values()))))
            rule_satisfactions = []

            # Sample node pairs for efficiency
            node_pairs = []
            for _ in range(500):  # Reduced for speed
                nodes = random.sample(list(self.graph.nodes())[:1000], 2)  # Use a subset of graph nodes
                node_pairs.append((nodes[0], nodes[1]))

            # Evaluate rules on each node pair
            for node_pair in node_pairs:
                node1, node2 = node_pair

                # Skip if embeddings not available
                if node1 not in self.verified_embeddings or node2 not in self.verified_embeddings:
                    continue

                # Get rule satisfactions for this pair
                satisfactions = self.evaluate_rules(node_pair)
                rule_satisfactions.extend(satisfactions)

                # Calculate overall rule satisfaction using weighted sum
                weighted_satisfaction = sum(w * s for w, s in zip(self.weights, satisfactions)) / sum(self.weights)

                # If rules are violated, adjust embeddings to reduce violation
                if weighted_satisfaction < 0.5:
                    # Move embeddings closer together if they should be similar
                    e1 = self.verified_embeddings[node1]
                    e2 = self.verified_embeddings[node2]

                    # Calculate adjustment vector - move towards or away based on satisfaction
                    direction = e2 - e1
                    direction_norm = np.linalg.norm(direction)

                    if direction_norm > 0:
                        adjustment = direction / direction_norm * learn_rate * (0.5 - weighted_satisfaction)
                        adjustments[node1] += adjustment
                        adjustments[node2] -= adjustment

            # Apply adjustments to embeddings
            for node, adjustment in adjustments.items():
                if node in self.verified_embeddings:
                    self.verified_embeddings[node] += adjustment
                    # Normalize to unit length
                    norm = np.linalg.norm(self.verified_embeddings[node])
                    if norm > 0:
                        self.verified_embeddings[node] /= norm

            # Update similarity matrix with new embeddings (periodically)
            if iteration % 5 == 0:
                self.embedding_sim = self._calculate_similarity()

                # Calculate overall rule satisfaction for logging
                if rule_satisfactions:
                    avg_satisfaction = sum(rule_satisfactions) / len(rule_satisfactions)
                    print(f"Iteration {iteration}: Average rule satisfaction: {avg_satisfaction:.4f}")

        print("Verification complete")
        return self.verified_embeddings

    def evaluate_verification(self):
        """Compare original and verified embeddings"""
        # Skip if verification hasn't been run
        if self.verified_embeddings is None:
            print("No verified embeddings available. Run verify_embeddings() first.")
            return {}

        # Calculate rule satisfaction before and after verification
        results = {}

        # Sample node pairs
        node_pairs = []
        for _ in range(500):  # Reduced for speed
            nodes = random.sample(list(self.graph.nodes())[:1000], 2)
            node_pairs.append((nodes[0], nodes[1]))

        # Save original embedding sim
        original_sim = self.embedding_sim.copy()

        # Calculate satisfaction with original embeddings
        orig_embeddings = self.embeddings
        self.embeddings = orig_embeddings
        self.embedding_sim = self._calculate_similarity()
        original_satisfactions = []
        for node_pair in node_pairs:
            if node_pair[0] in self.embeddings and node_pair[1] in self.embeddings:
                original_satisfactions.extend(self.evaluate_rules(node_pair))

        # Calculate satisfaction with verified embeddings
        self.embeddings = self.verified_embeddings
        self.embedding_sim = self._calculate_similarity()
        verified_satisfactions = []
        for node_pair in node_pairs:
            if node_pair[0] in self.embeddings and node_pair[1] in self.embeddings:
                verified_satisfactions.extend(self.evaluate_rules(node_pair))

        # Calculate average satisfaction
        if original_satisfactions and verified_satisfactions:
            results['original_avg_satisfaction'] = sum(original_satisfactions) / len(original_satisfactions)
            results['verified_avg_satisfaction'] = sum(verified_satisfactions) / len(verified_satisfactions)
            results['improvement'] = results['verified_avg_satisfaction'] - results['original_avg_satisfaction']
        else:
            results = {'original_avg_satisfaction': 0.5, 'verified_avg_satisfaction': 0.5, 'improvement': 0}

        # Restore state
        self.embeddings = orig_embeddings
        self.embedding_sim = original_sim

        return results


# If run directly, just show the class info
if __name__ == "__main__":
    print("Hybrid Markov Logic Network implementation for embedding verification")
    print("Available rules:")
    print("- citation_category: If A cites B, they likely have the same category")
    print("- co_citation_category: If A and B are both cited by C, they likely have the same category")
    print("- embedding_similarity: High embedding similarity implies same category")