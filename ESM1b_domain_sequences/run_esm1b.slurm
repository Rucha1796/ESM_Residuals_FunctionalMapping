#!/bin/bash
#SBATCH --job-name=esm1b_cpu_job
#SBATCH --partition=nodes             
#SBATCH --cpus-per-task=4             #Number of CPU cores
#SBATCH --mem=16G                     #Memory per node
#SBATCH --time=00:00:00              
#SBATCH --output=esm1b_cpu_out.log    
#SBATCH --error=esm1b_cpu_err.log    

echo "Activating environment..."
source /scratch/rdeo/ESM1b/esm1b_env/bin/activate

cd /scratch/rdeo/ESM1b/

echo "Running ESM1b CPU pipeline..."
python esm1b_main_dom.py \
  --input-csv-file domain_sequences_input.csv \
  --output-csv-file esm1b_scores_output.csv \
  --model-name esm1b_t33_650M_UR50S

echo "Job completed."
