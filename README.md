# Verification of Embeddings with Hybrid Markov Logic

This project demonstrates the application of Hybrid Markov Logic Networks (MLN) to verify and refine embeddings from the Cora citation network dataset.

## Project Files

- cora_preprocessing.py: Downloads and prepares the Cora dataset
- cora_embeddings.py: Generates embeddings for papers in the network
- hybrid_mln.py: Implements the Hybrid Markov Logic Network
- main_analysis.py: Runs the verification process and creates visualizations
- `run_analysis.py`: Main script to execute the entire workflow

## Requirements

- Python 3.7+
- NetworkX
- NumPy
- Pandas
- Matplotlib
- scikit-learn

```bash
pip install networkx==2.8.8 numpy==1.24.2 pandas==1.5.3 matplotlib==3.7.1 scikit-learn==1.2.2 node2vec==0.4.6 gensim==4.3.0
```

## Running the Analysis

```bash
python run_analysis.py
```

This will:
1. Download and preprocess the Cora dataset
2. Generate network embeddings
3. Apply Hybrid MLN rules for verification
4. Produce analysis results and visualizations

## Dataset

The Cora dataset is a citation network of computer science research papers:
- Contains approximately 2,708 scientific publications
- Categorized into 7 classes
- Each paper is represented by a 1433-dimensional binary feature vector
- The citation network contains 5,429 links

## Methodology

The project follows these steps:
1. **Preprocessing**: Clean and prepare the Cora citation network
2. **Embedding Generation**: Create vector representations of papers using node2vec
3. **Hybrid MLN Application**: Apply logical rules with embedding-based features
4. **Verification**: Analyze the consistency between citation patterns and embeddings
5. **Refinement**: Adjust embeddings based on logical constraints

## Output

Running the analysis will generate:
- Embedding visualization plots
- Verification statistics in CSV format
- Performance metrics of the hybrid approach
- Network visualizations colored by paper categories

## Visualization Examples

The output visualizations show:
- Paper clusters based on embeddings
- Citation patterns overlaid on embeddings
- Verification results highlighting inconsistencies

## Contact

For questions or feedback, please contact: zbairali7893@gmail.com