# Verification of Embeddings with Hybrid Markov Logic

This project demonstrates the application of Hybrid Markov Logic Networks (MLN) to verify and refine embeddings from the Cora citation network dataset.

## Project Files

- `cora_preprocessing.py`: Downloads and prepares the Cora dataset
- `cora_embeddings.py`: Generates embeddings for papers in the network
- `hybrid_mln.py`: Implements the Hybrid Markov Logic Network
- `main_analysis.py`: Runs the verification process and creates visualizations
- `run_analysis.py`: Main script to execute the entire workflow

## Requirements

- Python 3.7+
- NetworkX
- NumPy
- Pandas
- Matplotlib
- scikit-learn

pip install networkx==2.8.8 numpy==1.24.2 pandas==1.5.3 matplotlib==3.7.1 scikit-learn==1.2.2 node2vec==0.4.6 gensim==4.3.0

## Running the Analysis

```bash
python run_analysis.py