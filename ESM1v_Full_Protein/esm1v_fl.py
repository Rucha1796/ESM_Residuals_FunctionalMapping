# esm1v_fl_updated.py
import argparse
import pandas as pd
import torch
import numpy as np
from esm_variants_utils import load_esm_model, get_wt_LLR

AAorder = ['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']

def compute_summary_stats(logits_vector):
    max_val = np.max(logits_vector)
    mean_val = np.mean(logits_vector)
    exp_vals = np.exp(logits_vector)
    probs = exp_vals / np.sum(exp_vals)
    entropy_val = -np.sum(probs * np.log(probs + 1e-10))
    aa_max = AAorder[np.argmax(logits_vector)]
    return max_val, mean_val, entropy_val, aa_max

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {}.'.format('GPU' if device == 'cuda' else 'CPU'))

    # Read full protein CSV file
    input_df = pd.read_csv(args.input_csv_file, sep='\t')
    print(f"Before filtering: {len(input_df)} rows")
    input_df = input_df[input_df['wt_seq_fl'].notna() & input_df['wt_seq_fl'].apply(lambda x: isinstance(x, str))].copy()
    print(f"After filtering invalid sequences: {len(input_df)} rows")

    # Rename columns and keep both full protein sequence and domain sequence.
    # - 'seq' will be the full protein sequence (from wt_seq_fl) for model inference.
    # - 'domain_seq' (from wt_seq) is the domain sequence.
    # - 'length' is the domain length.
    input_df = input_df.rename(columns={
        'wt_seq_fl': 'seq',       # Full protein sequence (for model inference)
        'wt_seq': 'domain_seq',   # Domain sequence
        'dom_ID': 'id',
        'Gene_Names_(primary)': 'gene',
        'seq_length': 'length'
    })[['id', 'gene', 'seq', 'domain_seq', 'length']]

    print('Loading the model ({})...'.format(args.model_name))
    model, alphabet, batch_converter, repr_layer = load_esm_model(args.model_name, device)

    print('Invoking the model...')
    input_df_ids, results_list = get_wt_LLR(
        input_df, 
        model=model, 
        alphabet=alphabet, 
        batch_converter=batch_converter, 
        device=device, 
        return_wtlogits=True, 
        return_embedding=True
    )

    output_rows = []
    embedding_rows = []

    #Iterate over each protein
    for idx, (seq_id, (LLR, WTlogits, raw_logits_array, wt_embedding)) in enumerate(zip(input_df_ids, results_list)):
        #Get the corresponding row from the DataFrame to access domain info.
        current_row = input_df.iloc[idx]
        full_seq = current_row['seq']           # Full protein sequence
        domain_seq = current_row['domain_seq']    # Domain sequence
        domain_length = current_row['length']     # Domain length

        #Compute the domain start in the full protein (1-indexed)
        domain_start = full_seq.find(domain_seq) + 1
        domain_end = domain_start + domain_length - 1

        #Melt LLR into long format.
        melted = LLR.transpose().stack().reset_index().rename(
            columns={'level_0': 'wt_aa_and_pos', 'level_1': 'mut_aa', 0: 'esm_score'}
        )
        melted['seq_id'] = seq_id
        
        melted['pos'] = melted['wt_aa_and_pos'].apply(lambda x: int(x.split(' ')[1]))

        # Filter: only keep residues that lie within the domain boundaries
        melted = melted[(melted['pos'] >= domain_start) & (melted['pos'] <= domain_end)].copy()

        # Remap global positions to domain-local positions
        melted['local_pos'] = melted['pos'] - domain_start + 1

        #Compute per-position summary statistics
        pos_stats = {}
        for col in WTlogits.columns:
            p = int(col.split(' ')[1])
            if p < domain_start or p > domain_end:
                continue
            vec = WTlogits[col].values
            max_logit, mean_logit, entropy, aa_max = compute_summary_stats(vec)
            pos_stats[p] = (max_logit, mean_logit, entropy, aa_max)

        melted['max_logit'] = melted['pos'].apply(lambda p: pos_stats[p][0] if p in pos_stats else np.nan)
        melted['mean_logit'] = melted['pos'].apply(lambda p: pos_stats[p][1] if p in pos_stats else np.nan)
        melted['entropy'] = melted['pos'].apply(lambda p: pos_stats[p][2] if p in pos_stats else np.nan)
        melted['aa_max_logit'] = melted['pos'].apply(lambda p: pos_stats[p][3] if p in pos_stats else None)

        #Build mutation name using domain-local numbering.
        # Extract the wild-type residue from the domain sequence.
        melted['wt_res_domain'] = melted['local_pos'].apply(lambda lp: domain_seq[lp-1] if lp-1 < len(domain_seq) else '?')
        melted['mut_name'] = melted['seq_id'] + "_" + melted['wt_res_domain'] + melted['local_pos'].astype(str) + melted['mut_aa']

        #Retrieve raw logits and log probabilities.
        raw_logit_values = []
        log_prob_values = []
        for idx2, row in melted.iterrows():
            aa_idx = alphabet.tok_to_idx[row['mut_aa']]
            #Use global pos to index the raw logits array.
            raw_logit = raw_logits_array[row['pos'] - 1, aa_idx]
            raw_logit_values.append(raw_logit)
            log_prob_values.append(row['esm_score'])

        melted['raw_logit'] = raw_logit_values
        melted['log_prob'] = log_prob_values

        #rename local_pos to pos
        melted = melted.rename(columns={'local_pos': 'pos'})
        
        output_rows.append(melted[['seq_id', 'mut_name', 'pos', 'raw_logit', 'log_prob',
                                     'esm_score', 'max_logit', 'mean_logit', 'entropy', 'aa_max_logit']])

        #Append WT mean embedding as a new row.
        embedding_rows.append(pd.Series(wt_embedding, name=seq_id))

    results = pd.concat(output_rows).reset_index(drop=True)
    results.to_csv(args.output_csv_file, index=False)

    #Save the wild-type embeddings.
    embeddings_df = pd.DataFrame(embedding_rows)
    embeddings_df.index.name = 'seq_id'
    embeddings_df.to_csv(args.output_embedding_file)

    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute ESM scores + summary statistics from CSV.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input-csv-file', dest='input_csv_file', required=True, 
                        help='Input CSV file with wt_seq, dom_ID, Gene_Names_(primary), seq_length, wt_seq_fl')
    parser.add_argument('--output-csv-file', dest='output_csv_file', required=True, 
                        help='Output CSV for results')
    parser.add_argument('--output-embedding-file', dest='output_embedding_file', required=True, 
                        help='Output CSV file for WT mean embeddings')
    parser.add_argument('--model-name', dest='model_name', default='esm1b_t33_650M_UR50S', 
                        help='ESM model name')
    args = parser.parse_args()
    main(args)
