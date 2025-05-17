# 

This repository has all the code and datasets for the project Titled 'Combining ESM models with Experimentally Derived Structural Stability to Identify Functional Missense Mutations' completed In Partial Fulfillment of the Requirements for the Degree Master of Science at San Jose State University under the guidance of Dr. William Bill Andreopoulos. 


The effect of functional missense mutations is challenging to predict. The goal of this project is to to develop and apply integrated approaches that can systematically identify and interpret these hidden, functionally significant mutations. We Focus on identifying mutations that are structurally stable but have functional relevance. By doing so we aim to provide insights for prioritizing mutations and amino acid positions for experimental validation and clinical assessment.


 The pipeline includes:
- Inference (Zero-shot) on **domain-only sequences** using both ESM1b and ESM1v models.
- Inference on **full-length protein sequences** using ESM1v with a sliding window approach.
- Downstream correlation analysis 
- Functional analysis to identify potential **functional missense mutations**.
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
  
- .ipynb files from 00 to 04 contains the downstream analysis scripts for functional analysis and identificiation of mutations that are structurally stable but have functional relevance. 

Each folder includes logs, inference scripts, output CSVs, and SLURM scripts to run jobs on an HPC environment. All the python scripts for downstram analysis and residual analysis are also deposited in tthis repository. 

## How to Use

Ensure you have installed the required dependencies and have access to GPU or CPU nodes. Then run the appropriate script inside each directory (`*_main_dom.py` or `*_fl.py`) with the desired input file to perform zero-shot inference using ESM models.
