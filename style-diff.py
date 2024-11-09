import torch
from torch.nn import functional as F
import os
import time
import pickle
from model import GPT, GPTConfig
from contextlib import nullcontext
from util.tokenizer import tokenize_chord_list
from scipy.special import rel_entr
from util.music_util import search_chord_pg, c_wow, c_wow_plus, c_ws, ChordParserFSM, convert_to_harte
from util.cprog_util import get_harte_chord_pg, get_chord_pg, augment_chord_list

from util.model_util import get_model, get_probs, token_seq_prob, predict, sample_chord_symbol

max_token_len = 128
device = 'mps'


def get_novel_chord(model_pack, max_len=max_token_len, n_cont_chord=5,
                    start=None, kld_thres=4.0):
    """
    Predicts a chord progression jointly on two models.
    If output softmax similar (under a threshold), a common chord progression is formed.
    Else, the model stops to finish predicting the novel chord, and records the progression so far

    :return:
    """
    assert kld_thres > 0
    if start is None:
        start = ['[cls]', 'C', 'maj']
    m1, m2, encode, decode = model_pack
    has_novel_chord = False
    count_novel = 0
    kld_list = []

    def add_token_by_policy(i1, i2, prob1, prob2):
        kld = sum(rel_entr(prob1[0].tolist(), prob2[0].tolist()))
        kld_list.append(kld)
        # print('KL Divergence is: ', kld)
        novel = False
        if kld < kld_thres:
            # sample from fine-tuned model
            idx_next = sample_chord_symbol(prob1, decode)
            i1 = torch.cat((i1, idx_next), dim=1)
            i2 = torch.cat((i2, idx_next), dim=1)
            token = decode(idx_next[0].tolist())
            if token == '[cls]':
                raise Exception("wtf")
            # print(token)
        else:
            idx_next1 = sample_chord_symbol(prob1, decode)
            idx_next2 = sample_chord_symbol(prob2, decode)

            token1, token2 = decode(idx_next1[0].tolist())[0], decode(idx_next2[0].tolist())[0]
            # print(token1, token2)
            if token1 == '.' and token2 != '.':
                i2 = torch.cat((i2, idx_next2), dim=1)  # finish i2 first
            elif token2 == '.' and token1 != '.':
                i1 = torch.cat((i1, idx_next1), dim=1)  # finish i1 first
            else:
                novel = True
                i1 = torch.cat((i1, idx_next2), dim=1)
                i2 = torch.cat((i2, idx_next2), dim=1)

        return i1, i2, novel

    assert m1.config.block_size == m2.config.block_size
    sids = encode(start)
    x = (torch.tensor(sids, dtype=torch.long, device=device)[None, ...])
    idx1 = idx2 = x

    with torch.no_grad():
        with nullcontext():
            for j in range(max_len):
                if count_novel >= n_cont_chord:
                    break

                idx1_cond = idx1 if idx1.size(1) <= m1.config.block_size else idx1[:, -m1.config.block_size:]
                idx2_cond = idx2 if idx2.size(1) <= m1.config.block_size else idx2[:, -m1.config.block_size:]

                y1, _ = m1.forward(idx1_cond)
                y2, _ = m2.forward(idx2_cond)
                y1probs, y2probs = get_probs(y1), get_probs(y2)
                idx1, idx2, is_novel = add_token_by_policy(idx1, idx2, y1probs, y2probs)
                has_novel_chord |= is_novel
                if has_novel_chord and decode(idx1[0].tolist())[-1] == '.':
                    count_novel += 1

                # print(decode(idx1[0].tolist()))
                # print(decode(idx2[0].tolist()))
                # print('---------------')

    return decode(idx2[0].tolist()), kld_list


def compute_chord_joint_prob(seq, prob):
    """
    input a sequence of tokens and the corresponding probability, compute the joint prob of:
    - chord root & chord type
    :param seq:
    :param prob:
    :return:
    """
    out_chords = []
    out_probs = []
    tmp_chord, tmp_probs = [], []
    watch_count = 2
    for i, e in enumerate(seq):
        if e == '.':
            watch_count = 2  # pay attention to the following two tokens
            if tmp_probs:
                out_chords.append(tmp_chord)
                out_probs.append(tmp_probs)
                tmp_probs = []
                tmp_chord = []

        if watch_count > 0 and e != '.':
            tmp_chord.append(e)
            tmp_probs.append(prob[i])
            watch_count -= 1
    return out_chords, out_probs


def check_n_gram(prog_generated, prog_original, n_gram=2):
    """

    :param prog_generated: have to be [ [symbol, type], [s, t], ......]
    :param prog_original: same
    :param n_gram:
    :return:
    """
    out = []
    for i in range(len(prog_original) - n_gram):
        target_prog = prog_original[i: i + n_gram]
        pg_indice = search_chord_pg(prog_generated, target_prog)
        # print('prog: ', target_prog)
        # print("Indices:", pg_indice)
        out.append([target_prog, pg_indice])
    return out


def find_progression(base=c_wow, harte_prog_target=None):
    if harte_prog_target is None:
        harte_prog_target = ['F:min7/b3', 'G:maj/5', 'C:maj/3']
    return search_chord_pg(get_harte_chord_pg(base), get_harte_chord_pg(harte_prog_target))


def test_seqprob():
    m1, encode, decode = get_model('out-chords')
    m2, _, _ = get_model('out-chords-oworld')
    tokens = tokenize_chord_list(c_wow_plus)
    probs1 = token_seq_prob(tokens, m1, encode, decode)
    probs2 = token_seq_prob(tokens, m2, encode, decode)
    for i in range(len(tokens) - 1):
        print(tokens[i + 1], '\t', round(probs1[i], 3), round(probs2[i], 3))


def trim_incomplete_chord_seq(clist):
    """
    For sequences not ending on '.', or [cls], it could be C, maj, add, (missing info here)
    :param clist:
    :return:
    """
    if clist[-1] == '[cls]' or clist[-1] == '.':
        return clist
    else:
        nearest_end = len(clist) - 1
        while nearest_end > 0 and clist[nearest_end] != '.':
            nearest_end -= 1
        return clist[:nearest_end]


def get_chord_train_data(strategy=None):
    data = augment_chord_list(c_ws,
                              selected_prog=[['A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7']],
                              strategy=strategy)
    out = []

    cp = ChordParserFSM()
    for clist in data:
        for e in clist:
            cp.reset()
            out.append(cp.tokenize(e)[:2])
        out.append('[cls]')
    return out


def get_ngram_stats(og_prog, pred_prog, n_grams=None):
    if n_grams is None:
        n_grams = [2]
    grams_output = []
    for i in n_grams:
        grams_output.append(check_n_gram(pred_prog, og_prog, n_gram=i))
    ngram_stats = {}
    distributions = []
    symbols = []
    for i, ng_result in enumerate(grams_output):
        dist = []
        sym = []
        for e in ng_result:
            sym.append(e[0])
            dist.append(len(e[1]))
        distributions.append(dist)
        symbols.append(sym)
    for i in range(len(n_grams)):
        assert len(distributions[i]) == len(symbols[i])
        ngram_stats[n_grams[i]] = [symbols[i], distributions[i]]
    return ngram_stats


def print_ngram_states(stats):
    print("===== N-gram stats =====")
    for k in stats.keys():
        print("{}-gram:".format(k))
        for j in range(len(stats[k][0])):
            print("Symbol: ", stats[k][0][j], stats[k][1][j])


def test_ngram(n_preds=10, n_grams=None):
    if n_grams is None:
        n_grams = [2, 3, 4]
    prog = get_harte_chord_pg(c_ws)

    model, encode, decode = get_model('out-chords-ws-selectgen')
    # model, encode, decode = get_model('out-chords-oworld')
    output = []
    for i in range(n_preds):
        # start = time.time()
        predicted_seq = predict(model, encode, decode, start=['[cls]', ], max_seq_len=60)
        # print_ngram_states(get_ngram_stats(og_prog=prog, pred_prog=get_chord_pg(predicted_seq), n_grams=n_grams))
        output.extend(trim_incomplete_chord_seq(predicted_seq))
        output.append('[cls]')
        # end = time.time()
        # print('Time for', i, 'th prediction:', end - start)
    pred_chord_pg = get_chord_pg(output)
    # pred_chord_pg = get_chord_train_data(strategy='select-gen')

    stats = get_ngram_stats(og_prog=prog, pred_prog=pred_chord_pg, n_grams=n_grams)
    print_ngram_states(stats)


def test_joint_pred(m1folder='out-chords', m2folder='out-chords-ws-raw', start=None, kld_thres=2.0, n_cont_chord=5,
                    n_tries=1):
    if start is None:
        start = ['[cls]']
    m1, encode, decode = get_model(m1folder)
    m2, _, _ = get_model(m2folder)
    ## Jointly predicts until KL divergence is significant

    pred_tk_list = []
    kld_list = []
    for i in range(n_tries):
        tmp_klds = []
        out, kld = get_novel_chord((m1, m2, encode, decode),
                                   start=start,
                                   kld_thres=kld_thres,
                                   n_cont_chord=n_cont_chord)
        pred_tk_list.extend(trim_incomplete_chord_seq(out))
        # for j in range(len(start)):
        #     print(start[j], '\t', 'init')
        for j in range(len(out) - len(start)):
            if out[j + len(start)] not in ['.', '/']:
                tmp_klds.append(kld[j])
            # print(out[j + len(start)], '\t', "\x1B[33m{}\033[0m".format(kld[j]) if kld[j] > kld_thres else kld[j])
        kld_list.append(sum(tmp_klds) / len(tmp_klds))
    print("Average total kld:", sum(kld_list) / len(kld_list))

    og_prog = get_harte_chord_pg(c_ws)
    pred_prog = get_chord_pg(pred_tk_list)
    print_ngram_states(get_ngram_stats(og_prog, pred_prog, n_grams=[2, 3, 4]))


if __name__ == '__main__':
    # m1, encode, decode = get_model('out-chords')
    # m2, _, _ = get_model('out-chords-ws-selectkmp')

    # test_seqprob()
    # test_ngram(n_grams=[2, 3, 4], n_preds=100)
    kld = []
    test_joint_pred(m2folder='out-chords-ws-raw', n_cont_chord=15, kld_thres=1, n_tries=100)
    # for k in range(1, 9):
    #     kld.append([k / 2, test_joint_pred(m2folder='out-chords-ws-selectgen', n_cont_chord=8, kld_thres=k / 2)])
    # for e in kld:
    #     print('kld:', e)

    # tc, tp = compute_chord_joint_prob(wow_tokens, probs2)
    # for i in range(len(tc)):
    #     print(tc[i], tp[i])

    # print("\x1B[31m[Error]\033[0m ")
