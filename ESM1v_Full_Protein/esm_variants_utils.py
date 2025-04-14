import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

AAorder = ['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_esm_model(model_name, device=0):
    repr_layer = int(model_name.split('_')[1][1:])
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    batch_converter = alphabet.get_batch_converter()
    return model.eval().to(device), alphabet, batch_converter, repr_layer

def get_wt_LLR(input_df, model, alphabet, batch_converter, device=0, silent=False,
               return_wtlogits=False, return_embedding=False):
    genes = input_df.id.values
    LLRs = []
    input_df_ids = []
    log_probs_list = []
    raw_logits_list = []
    wt_embeddings_list = []

    for gname in tqdm(genes, disable=silent):
        seq = input_df[input_df.id == gname].seq.values[0]
        seq_length = len(seq)

        if seq_length <= 1022:
            dt = [(gname + '_WT', seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(dt)

            with torch.no_grad():
                output = model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
                logits = output['logits'][0, 1:-1, :].cpu().numpy()
                token_representations = output["representations"][33][0, 1:-1]
                mean_embedding = token_representations.mean(0).cpu().numpy()

            log_probs = torch.log_softmax(output['logits'], dim=-1)[0, 1:-1, :].cpu().numpy()

        else:
            ints, M, M_norm = get_intervals_and_weights(seq_length)
            tokens = []
            for i, idx in enumerate(ints):
                sub_seq = ''.join(np.array(list(seq))[idx])
                tokens.append((f"{gname}_WT_{i}", sub_seq))

            logits_chunks = []
            representations_chunks = []

            for batch in chunks(tokens, 20):
                batch_labels, batch_strs, batch_tokens = batch_converter(batch)
                with torch.no_grad():
                    output = model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
                    logits_batch = output['logits'][:, 1:-1, :].cpu().numpy()
                    reprs_batch = output['representations'][33][:, 1:-1, :].cpu()

                for i in range(len(batch)):
                    logits_chunks.append(logits_batch[i])
                    representations_chunks.append(reprs_batch[i])

            logits = np.zeros((seq_length, logits_chunks[0].shape[1]))
            reprs = torch.zeros((seq_length, representations_chunks[0].shape[1]))

            for i in range(len(ints)):
                interval = ints[i]
                logit_slice = logits_chunks[i]
                repr_slice = representations_chunks[i]
                weight_slice = M_norm[i]

                min_len = min(len(interval), logit_slice.shape[0], weight_slice.shape[0])
                logits[interval[:min_len]] += logit_slice[:min_len] * weight_slice[:min_len][:, None]
                reprs[interval[:min_len]] += repr_slice[:min_len] * torch.tensor(weight_slice[:min_len][:, None], device=repr_slice.device)

            mean_embedding = reprs.mean(0).cpu().numpy()
            log_probs = torch.log_softmax(torch.tensor(logits), dim=-1).numpy()

        log_prob_df = pd.DataFrame(
            log_probs,
            columns=alphabet.all_toks,
            index=list(seq)
        ).T.iloc[4:24].loc[AAorder]
        log_prob_df.columns = [aa + ' ' + str(i+1) for i, aa in enumerate(seq)]

        wt_norm = np.diag(log_prob_df.loc[[aa.split(' ')[0] for aa in log_prob_df.columns]])
        LLR = log_prob_df - wt_norm

        LLRs.append(LLR)
        input_df_ids.append(gname)

        if return_wtlogits or return_embedding:
            log_probs_list.append(log_prob_df)
            raw_logits_list.append(logits)
            wt_embeddings_list.append(mean_embedding)

    if return_embedding:
        return input_df_ids, list(zip(LLRs, log_probs_list, raw_logits_list, wt_embeddings_list))
    elif return_wtlogits:
        return input_df_ids, list(zip(LLRs, log_probs_list, raw_logits_list))
    else:
        return input_df_ids, LLRs

#Sliding window utilities

def chop(L, min_overlap=511, max_len=1022):
    return L[max_len - min_overlap : -max_len + min_overlap]

def intervals(L, min_overlap=511, max_len=1022, parts=None):
    if parts is None:
        parts = []
    if len(L) <= max_len:
        if parts and (parts[-2][-1] - parts[-1][0] < min_overlap):
            return parts + [np.arange(L[int(len(L)/2)] - int(max_len/2), L[int(len(L)/2)] + int(max_len/2))]
        else:
            return parts
    else:
        parts += [L[:max_len], L[-max_len:]]
        L = chop(L, min_overlap, max_len)
        return intervals(L, min_overlap, max_len, parts=parts)

def get_intervals_and_weights(seq_len, min_overlap=511, max_len=1022, s=16):
    ints = intervals(np.arange(seq_len), min_overlap=min_overlap, max_len=max_len)
    ints = [ints[i] for i in np.argsort([i[0] for i in ints])]
    a = int(np.round(min_overlap / 2))
    t = np.arange(max_len)
    f = np.ones(max_len)
    f[:a] = 1 / (1 + np.exp(-(t[:a] - a/2)/s))
    f[max_len - a:] = 1 / (1 + np.exp((t[:a] - a/2)/s))
    f0 = np.ones(max_len)
    f0[max_len - a:] = 1 / (1 + np.exp((t[:a] - a/2)/s))
    fn = np.ones(max_len)
    fn[:a] = 1 / (1 + np.exp(-(t[:a] - a/2)/s))
    filt = [f0] + [f for _ in ints[1:-1]] + [fn]
    M = np.zeros((len(ints), seq_len))
    for k, i in enumerate(ints):
        M[k, i] = filt[k][:len(i)]  # ensure length matches
    M_norm = M / M.sum(0)
    return (ints, M, M_norm)
