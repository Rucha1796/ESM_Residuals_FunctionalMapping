# ESM Residual‑Based Functional Mutation Analysis

This repository benchmarks two protein language models—ESM1b and ESM1v—against experimental fitness data and uses a residual‑based workflow to flag candidate functional missense mutations. The pipeline includes:

- Inference on **domain-only sequences** using both ESM1b and ESM1v.
- Inference on **full-length protein sequences** using ESM1v with a sliding window approach.
- Downstream correlation analysis 
- Residual-based analysis to identify potential **functional missense mutations**.
- Structural mapping using **AlphaFold2 PDBs** and clustering of high-residual sites.

## Folders Overview

- `ESM1b_domain_sequences/`  
  Contains scripts, outputs and input dataset for ESM1b domain-level inference.

- `ESM1v_domain_sequences/`  
  Contains scripts, outputs and input dataset for ESM1v domain-level inference.

- `ESM1v_Full_Protein/`  
  Contains scripts and output and input dataset for ESM1v inference on full-length sequences using a sliding window strategy.

- `alphafold_structures.zip`  
  Includes PDB files used for structural mapping of high-residual mutation sites.

Each folder includes logs, inference scripts, output CSVs, and SLURM scripts to run jobs on an HPC environment. All the python scripts for downstram analysis and residual analysis are also deposited in tthis repository. 

## How to Use

Ensure you have installed the required dependencies and have access to GPU or CPU nodes. Then run the appropriate script inside each directory (`*_main_dom.py` or `*_fl.py`) with the desired input file to perform zero-shot inference using ESM models.
