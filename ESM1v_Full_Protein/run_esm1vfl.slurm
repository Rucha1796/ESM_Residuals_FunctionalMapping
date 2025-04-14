#!/bin/bash
#SBATCH --job-name=esm1v_embed_job
#SBATCH --partition=nodes             # Use CPU partition (adjust if needed)
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --mem=16G                     # Memory per node
#SBATCH --time=09:00:00               # Job time limit (adjust as needed)
#SBATCH --output=esm1v_embed_out.log  # STDOUT log
#SBATCH --error=esm1v_embed_err.log   # STDERR log

echo "Activating environment..."
source /scratch/rdeo/ESM1v/esm1v_new_env/bin/activate

cd /scratch/rdeo/ESM1v/ESM1v_flPotSeq

echo "Running ESM1v CPU pipeline with WT embeddings..."
python esm1v_fl.py \
  --input-csv-file ESM1v_full_with_wt_seq_fl.csv \
  --output-csv-file esm1v_scores_outputfl.csv \
  --output-embedding-file esm1v_wt_embeddingsfl.csv \
  --model-name esm1v_t33_650M_UR90S

echo "Job completed."
