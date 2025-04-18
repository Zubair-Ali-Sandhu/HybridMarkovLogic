import os
import sys
import importlib

# List of needed packages
required_packages = [
    'networkx', 'numpy', 'pandas', 'matplotlib', 'scikit-learn'
]


def check_and_install_packages():
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")

    # Check for node2vec separately - not required since we're using spectral embeddings
    try:
        importlib.import_module('node2vec')
    except ImportError:
        print("Note: node2vec not installed, but not required for this analysis.")


print("Checking required packages...")
check_and_install_packages()

print("Starting Cora dataset analysis for embedding verification with Hybrid MLN")

# Make sure each script can run independently
print("\n1. Preprocessing Cora dataset...")
exec(open("cora_preprocessing.py").read())

print("\n2. Generating node embeddings...")
exec(open("cora_embeddings.py").read())

print("\n3. Running main analysis and visualization...")
exec(open("main_analysis.py").read())

print("\nAll analysis complete!")
print("Generated files for your presentation:")
for file in os.listdir():
    if file.endswith(".png") or file == "mln_rules.txt":
        print(f" - {file}")