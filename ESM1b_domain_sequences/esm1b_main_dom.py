# esm_score_missense_mutations_csv.py
import argparse
import pandas as pd
import torch
import numpy as np
from esm_variants_utils import load_esm_model, get_wt_LLR

def compute_summary_stats(logits_vector):
    max_val = np.max(logits_vector)
    mean_val = np.mean(logits_vector)
    exp_vals = np.exp(logits_vector)
    probs = exp_vals / np.sum(exp_vals)
    entropy_val = -np.sum(probs * np.log(probs + 1e-10))
    return max_val, mean_val, entropy_val, np.argmax(logits_vector)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {}.'.format('GPU' if device == 'cuda' else 'CPU (this may be much slower)'))

    input_df = pd.read_csv(args.input_csv_file)
    input_df = input_df.rename(columns={
        'wt_seq': 'seq',
        'dom_ID': 'id',
        'Gene_Names_(primary)': 'gene',
        'seq_length': 'length'
    })[['id', 'gene', 'seq', 'length']]

    print('Loading the model ({})...'.format(args.model_name))
    model, alphabet, batch_converter, repr_layer = load_esm_model(args.model_name, device)

    print('Invoking the model...')
    input_df_ids, results_list = get_wt_LLR(input_df, model=model, alphabet=alphabet, batch_converter=batch_converter, device=device, return_wtlogits=True)

    output_rows = []
    for seq_id, (LLR, WTlogits, raw_logits) in zip(input_df_ids, results_list):
        melted = LLR.transpose().stack().reset_index().rename(columns={'level_0': 'wt_aa_and_pos', 'level_1': 'mut_aa', 0: 'esm_score'})
        melted['seq_id'] = seq_id
        melted['pos'] = melted['wt_aa_and_pos'].apply(lambda x: int(x.split(' ')[1]))

        pos_stats = {}
        aa_max_logit = {}
        for col in WTlogits.columns:
            p = int(col.split(' ')[1])
            vec = WTlogits[col].values
            max_val, mean_val, entropy_val, max_idx = compute_summary_stats(vec)
            max_aa = WTlogits.index[max_idx]
            pos_stats[p] = (max_val, mean_val, entropy_val)
            aa_max_logit[p] = max_aa

        melted['max_logit'] = melted['pos'].apply(lambda p: pos_stats[p][0])
        melted['mean_logit'] = melted['pos'].apply(lambda p: pos_stats[p][1])
        melted['entropy'] = melted['pos'].apply(lambda p: pos_stats[p][2])
        melted['aa_max_logit'] = melted['pos'].apply(lambda p: aa_max_logit[p])

        # Get log prob and raw logit for the mutated amino acid
        for i, row in melted.iterrows():
            mut_aa = row['mut_aa']
            pos = row['pos'] - 1  # adjust for 0-based indexing in array
            melted.at[i, 'mut_log_prob'] = WTlogits.iloc[:, pos].loc[mut_aa]
            melted.at[i, 'mut_raw_logit'] = raw_logits[pos][alphabet.tok_to_idx[mut_aa]]

        # New formatted mut_name for alignment with Domainome
        melted['mut_name'] = melted['seq_id'] + "_" + \
                             melted['wt_aa_and_pos'].str[0] + \
                             melted['pos'].astype(str) + \
                             melted['mut_aa']

        output_rows.append(melted[['seq_id', 'mut_name', 'pos', 'esm_score', 'mut_log_prob', 'mut_raw_logit', 'max_logit', 'aa_max_logit', 'mean_logit', 'entropy']])

    results = pd.concat(output_rows).reset_index(drop=True)
    results.to_csv(args.output_csv_file, index=False)
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute ESM scores + summary statistics from CSV.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-csv-file', dest='input_csv_file', required=True, help='Input CSV file with wt_seq, dom_ID, Gene_Names_(primary), seq_length')
    parser.add_argument('--output-csv-file', dest='output_csv_file', required=True, help='Output CSV for results')
    parser.add_argument('--model-name', dest='model_name', default='esm1b_t33_650M_UR50S', help='ESM model name')
    args = parser.parse_args()
    main(args)
