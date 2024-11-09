import random

import torch
from torch.nn import functional as F
import os
import pickle
from model import GPT, GPTConfig
from contextlib import nullcontext
from util.tokenizer import tokenize_chord_list
from scipy.special import rel_entr
from util.music_util import search_chord_pg, pitch_names
import numpy as np
import itertools

device = 'mps'


def sample_chord_symbol(prob, decode, max_retries=10):
    idx_next = torch.multinomial(prob, num_samples=1)
    dec = decode(idx_next[0].tolist())[0]
    while dec == '[cls]' and max_retries >= 0:
        # Do not start a new piece with cls!
        idx_next = torch.multinomial(prob, num_samples=1)
        dec = decode(idx_next[0].tolist())[0]
        max_retries -= 1
    if max_retries <= 0:
        print('Max retries reached yet non-cls token is not sampled')
    return idx_next


def get_model(folder_path):
    cwd = ''
    if folder_path[0] == '/':
        cwd = folder_path.rsplit('/', 1)[0] + '/'
    ckpt_path = os.path.join(folder_path, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    assert 'config' in checkpoint and 'dataset' in checkpoint['config']
    meta_path = os.path.join(cwd + 'data', checkpoint['config']['dataset'], 'meta.pkl')
    assert os.path.exists(meta_path)

    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: [itos[i] for i in l]
    return model, encode, decode


def load_from_paths(ckpt_path, tokenizer_path, device_=None):
    if device_ is None:
        device_ = 'cpu'
    checkpoint = torch.load(ckpt_path, map_location=device_)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device_)
    assert 'config' in checkpoint and 'dataset' in checkpoint['config']
    assert os.path.exists(tokenizer_path)

    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: [itos[i] for i in l]
    return model, encode, decode


def get_probs(logits, temperature=1.0, top_k=200):
    new_lgts = logits[:, -1, :] / temperature
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, _ = torch.topk(new_lgts, min(top_k, new_lgts.size(-1)))
        new_lgts[new_lgts < v[:, [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    return F.softmax(new_lgts, dim=-1)


def token_seq_prob(seq, model: GPT, encode, decode):
    """
    Input a sequence of tokens, output probability for each token given gpt model
    :param seq:
    :param model:
    :return:
    """

    sids = encode(seq)
    x = (torch.tensor(sids, dtype=torch.long, device=device)[None, ...])
    probs = []
    assert x.size(1) >= 1
    with torch.no_grad():
        with nullcontext():
            for j in range(1, x.size(1)):
                idx = x[:, :j]
                idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

                y, _ = model.forward(idx_cond)
                y_probs = get_probs(y)

                next_token = x[:, j].item()
                probs.append(y_probs[0][next_token].item())

    return probs


def predict(model, encode, decode, start=None, temperature=1.0, top_k=200, max_seq_len=250, device='mps'):
    if start is None:
        start = ['[cls]']
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with nullcontext():
            y = model.generate(x, max_seq_len, temperature=temperature, top_k=top_k)
            return decode(y[0].tolist())


def get_next_chord_token(model, encode, decode, input=None, temperature=1.0, top_k=200, device='mps'):
    if input is None:
        input = ['[cls]']
    start_ids = encode(input)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with nullcontext():
            logits, _ = model.forward(x)
            probs = get_probs(logits, temperature=temperature, top_k=top_k)
            next_token = sample_chord_symbol(probs, decode)
            return decode(next_token[0].tolist())[0], probs[:, next_token[0].item()].item()


def gen_next_chord(model, encode, decode, input=None, temperature=1.0, top_k=200, device='mps'):
    """
    Generate a chord (returned as tokenized list)
    :return chord, probability (average of all p)
    """
    if input is None:
        input = ['[cls]']
    chord = []
    input_token_seq = input.copy()
    max_loop = 10
    chord_prob = []
    while max_loop > 0:
        tok, p = get_next_chord_token(model, encode, decode, input_token_seq, temperature=temperature, top_k=top_k,
                                      device=device)
        if tok == '[cls]':
            if not chord:
                chord_prob.append(p)
                chord.append(tok)
            break
        elif tok == '.' and chord:
            chord.append(tok)
            break
        input_token_seq.append(tok)
        chord_prob.append(p)
        chord.append(tok)
        max_loop -= 1
    return chord, sum(chord_prob) / len(chord_prob)


def iter_per_epoch(total_token_num, grad_acc_step, batch_sz, block_sz, ddp_world_size=1):
    return total_token_num / (grad_acc_step * batch_sz * block_sz * ddp_world_size)


def hits(hits_k: list, eval_tokens, model, temperature=1.0):
    """

    :param hits_k: number of top predictions to consider
    :param eval_tokens: evaluation tokens
    :param model: the GPT model
    :return:
    """
    assert type(hits_k) == list
    hits_k.sort()
    model.eval()
    hits = np.array([0] * len(hits_k))
    total = 0
    x = eval_tokens
    assert x.size(1) >= 1
    with torch.no_grad():
        with nullcontext():
            for j in range(1, x.size(1)):
                idx = x[:, :j]
                idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

                outputs, _ = model.forward(idx_cond)
                logits = outputs[:, -1, :] / temperature
                top_k = torch.topk(logits, hits_k[-1], dim=-1).indices.squeeze(0)
                for ji in range(x.size(0)):
                    # if x[ji, j].item() == '.' or x[ji, j].item() == "[cls]":
                    #     continue
                    for ik, nk in enumerate(hits_k):
                        if x[ji, j].item() in top_k[ji][:nk].tolist():
                            hits[ik] += 1
                    total += 1

    hits_at_k_score = hits / total
    return hits_at_k_score


def get_data(split, data_dir, block_size, batch_size):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def get_eval_tokens(data_dir, block_size, batch_size=64):
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    tokens = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    tokens = tokens.to(device)
    return tokens


def test_hits():
    # y = get_eval_tokens('/Users/kurono/Documents/python/GEC/research/data/chords',
    #                     256)
    # m1, encode, decode = get_model('/Users/kurono/Documents/python/GEC/research/out-chords')
    # print(decode(y[0].tolist()))
    # print(hits(3, y, m1))

    y = get_eval_tokens('/Users/kurono/Documents/python/GEC/research/data/chords-ws-raw',
                        256, batch_size=64)
    model, encode, decode = get_model('/Users/kurono/Documents/python/GEC/research/out-chords-ws-selectgenrep')
    print(decode(y[0].tolist()))
    print(hits([1, 3, 5], y, model))


def gen_chord_context(reversed_start_seq, model, encode, decode, n_candidate=1, n_recurse=5, max_recurse=0, stop='.'):
    """

    :param reversed_start_seq:
    :param model:
    :param encode:
    :param decode:
    :param n_candidate:   routes / splits per recursion
    :param n_recurse:     maximum token seq length (or recursion depth)
    :param max_recurse:   maximum recursions allowed to go over n_recurse
    :param stop:          stopping symbol - will continue to recurse unless it finds the stopping symbol.
    :return:
    """
    if n_recurse + max_recurse < 0:
        return [reversed_start_seq]
    elif n_recurse <= 0 and reversed_start_seq[0] == '.':
        return [reversed_start_seq]
    out = [reversed_start_seq]
    for i in range(n_candidate):
        new_seq = reversed_start_seq.copy()
        next_tok, _ = get_next_chord_token(model, encode, decode, reversed_start_seq)
        new_seq.append(next_tok)
        out.extend(
            gen_chord_context(new_seq, model, encode, decode,
                              n_candidate=n_candidate - 1 if n_candidate > 1 else n_candidate, n_recurse=n_recurse - 1,
                              max_recurse=max_recurse, stop=stop))
    return out


def gen_chord_context_with_replace(start_seq, replace_seq, model, encode, decode,
                                   n_candidate=1, n_recurse=5, max_recurse=0,
                                   stop='.'):
    """

    :param reversed_start_seq:
    :param model:
    :param encode:
    :param decode:
    :param n_candidate:   routes / splits per recursion
    :param n_recurse:     maximum token seq length (or recursion depth)
    :param max_recurse:   maximum recursions allowed to go over n_recurse
    :param stop:          stopping symbol - will continue to recurse unless it finds the stopping symbol.
    :return:
    """
    if n_recurse + max_recurse < 0:
        return [start_seq], [replace_seq]
    elif n_recurse <= 0 and replace_seq[0] == '.':
        return [start_seq], [replace_seq]
    out, rep_out = [start_seq], [replace_seq]
    for i in range(n_candidate):
        new_seq = start_seq.copy()
        rep_seq = replace_seq.copy()
        next_tok, _ = get_next_chord_token(model, encode, decode, rep_seq)
        new_seq.append(next_tok)
        rep_seq.append(next_tok)
        o, r = gen_chord_context_with_replace(new_seq, rep_seq, model, encode, decode,
                                              n_candidate=n_candidate - 1 if n_candidate > 1 else n_candidate,
                                              n_recurse=n_recurse - 1,
                                              max_recurse=max_recurse, stop=stop)
        out.extend(o)
        rep_out.extend(r)
    return out, rep_out


def filter_chord_context(ctx_list: list, min_length=9):
    """
    gen_chord_context creates a list of tokens. Some of which contains incomplete chord token.
    :param ctx_list:
    :param min_length: approximately 3 tokens per chord.
    :return:
    """
    assert min_length >= 1
    processed_ctx_list = ctx_list.copy()
    processed_ctx_list.sort()
    processed_ctx_list = list(k for k, _ in itertools.groupby(processed_ctx_list.copy()))
    for e in processed_ctx_list:
        e.reverse()

    ctx_out = []
    for e in processed_ctx_list:
        while '[cls]' in e:
            e.remove('[cls]')
        if len(e) < min_length:
            continue
        if e[0] == '.':
            ctx_out.append(e[1:])
        elif e[0] in pitch_names:
            ctx_out.append(e)
    return ctx_out


def cal_ctx_recurse_num(n_rec, n_cand):
    if n_rec <= 0:
        return 1
    sum = 1
    for i in range(n_cand):
        sum += cal_ctx_recurse_num(n_rec - 1, n_cand - 1 if n_cand > 1 else n_cand)
    return sum


if __name__ == '__main__':
    print(iter_per_epoch(305020, 1, 64, 256))
    # test_hits()

    # # print(gen_next_chord(model, encode, decode, input=['C', 'maj']))
    # md, encode, decode = get_model('/Users/kurono/Documents/python/GEC/research/out-chords')
    #
    # rev_model, encode, decode = get_model('/Users/kurono/Documents/python/GEC/research/out-revchords')
    # chord_ctx = ['G', 'maj', '.', 'C', 'maj', '[cls]']
    # replace = ['Ab', 'maj', '.', 'Db', 'maj', '[cls]']
    # chord_ctx.reverse()
    # replace.reverse()
    # n_recurse = 1
    # n_candidate = 1
    # # print("Approximate recursion num:", cal_ctx_recurse_num(n_recurse, n_candidate))
    # output, rev = gen_chord_context_with_replace(chord_ctx, replace, rev_model, encode, decode,
    #                                              n_recurse=n_recurse,
    #                                              max_recurse=5,
    #                                              n_candidate=n_candidate)
    # print(filter_chord_context(output, 1))
    # print(filter_chord_context(rev, 1))
    #
    # for i in filter_chord_context(output):
    #     print(i)
