"""
Create chord training data
"""
import numpy as np
import os
import pickle
from util.tokenizer import tokenize_dataset


def generate_data():
    tokens = tokenize_dataset()
    chars = sorted(list(set(tokens)))
    vocab_size = len(chars)
    print("all the unique characters:", ', '.join(chars))
    print(f"vocab size: {vocab_size:,}")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    def decode(l):
        return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    n = len(tokens)
    train_data = tokens[:int(n * 0.9)]
    val_data = tokens[int(n * 0.9):]

    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }

    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)


if __name__ == '__main__':
    # cp = ChordParserFSM()
    # print(cp.tokenize("Bb:7(b13)"))
    generate_data()

    '''
    all the unique characters: 
        ##1, ##4, #1, #11, #13, #3, #4, #5, #6, #7, #9, *1, *11, *3, *5, *b3, ., /, 
        1, 11, 13, 2, 3, 4, 5, 6, 7, 9,
        A, Ab, B, Bb, C, C#, Cb, D, Db, E, Eb, F, F#, Fb, G, G#, Gb, [cls],
        add, aug, b1, b13, b2, b3, b4, b5, b6, b7, b8, b9, bb3, bb7, bb9, 
        dim, dim7, hdim7, maj, maj6, maj7, maj9, min, min6, min7, min9, minmaj7, sus2, sus4
    vocab size: 75
    train has 73,896,826 tokens
    val has 8,210,759 tokens
    '''
