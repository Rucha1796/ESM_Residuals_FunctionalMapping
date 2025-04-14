import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

AAorder = ['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

##### INFERENCE
def load_esm_model(model_name, device=0):
    repr_layer = int(model_name.split('_')[1][1:])
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    batch_converter = alphabet.get_batch_converter()
    return model.eval().to(device), alphabet, batch_converter, repr_layer

def get_wt_LLR(input_df, model, alphabet, batch_converter, device=0, silent=False, return_wtlogits=False):
    genes = input_df.id.values
    LLRs = []
    input_df_ids = []
    log_probs_list = []
    raw_logits_list = []

    for gname in tqdm(genes, disable=silent):
        seq = input_df[input_df.id == gname].seq.values[0]
        seq_length = len(seq)

        dt = [(gname + '_WT', seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(dt)

        with torch.no_grad():
            output = model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
            raw_logits = output['logits'][0, 1:-1, :].cpu().numpy()
            log_probs = torch.log_softmax(output['logits'], dim=-1)[0, 1:-1, :].cpu().numpy()

        log_prob_df = pd.DataFrame(
            log_probs,
            columns=alphabet.all_toks,
            index=list(seq)
        ).T.iloc[4:24].loc[AAorder]
        log_prob_df.columns = [j.split('.')[0] + ' ' + str(i+1) for i, j in enumerate(log_prob_df.columns)]

        wt_norm = np.diag(log_prob_df.loc[[i.split(' ')[0] for i in log_prob_df.columns]])
        LLR = log_prob_df - wt_norm

        LLRs.append(LLR)
        input_df_ids.append(gname)

        if return_wtlogits:
            log_probs_list.append(log_prob_df)
            raw_logits_list.append(raw_logits)

    if return_wtlogits:
        return input_df_ids, list(zip(LLRs, log_probs_list, raw_logits_list))
    else:
        return input_df_ids, LLRs

def get_logits(seq, model, batch_converter, format=None, device=0):
    data = [("_", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        logits = torch.log_softmax(model(batch_tokens, repr_layers=[33], return_contacts=False)["logits"], dim=-1).cpu().numpy()
    if format == 'pandas':
        WTlogits = pd.DataFrame(logits[0, 1:-1, :], columns=alphabet.all_toks, index=list(seq)).T.iloc[4:24].loc[AAorder]
        WTlogits.columns = [j.split('.')[0] + ' ' + str(i+1) for i, j in enumerate(WTlogits.columns)]
        return WTlogits
    else:
        return logits[0, 1:-1, :]

def get_PLL(seq, model, alphabet, batch_converter, reduce=np.sum, device=0):
    s = get_logits(seq, model=model, batch_converter=batch_converter, device=device)
    idx = [alphabet.tok_to_idx[i] for i in seq]
    return reduce(np.diag(s[:, idx]))


##################### TILING utils ###########################

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
    a = int(np.round(min_overlap/2))
    t = np.arange(max_len)
    f = np.ones(max_len)
    f[:a] = 1 / (1 + np.exp(-(t[:a] - a/2)/s))
    f[max_len - a:] = 1 / (1 + np.exp((t[:a] - a/2)/s))
    f0 = np.ones(max_len)
    f0[max_len - a:] = 1 / (1 + np.exp((t[:a] - a/2)/s))
    fn = np.ones(max_len)
    fn[:a] = 1 / (1 + np.exp(-(t[:a] - a/2)/s))
    filt = [f0] + [f for i in ints[1:-1]] + [fn]
    M = np.zeros((len(ints), seq_len))
    for k, i in enumerate(ints):
        M[k, i] = filt[k]
    M_norm = M / M.sum(0)
    return (ints, M, M_norm)

