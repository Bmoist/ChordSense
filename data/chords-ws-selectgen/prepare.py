import numpy as np
import os
import pickle
import random
import shutil

from util.tokenizer import mod_and_tokenize_chords
from util.music_util import c_ws
from util.pickle_util import get_vocab_dict
from util.cprog_util import augment_chord_list

basedata_path = '../chords/meta.pkl'


# Hand-crafted list of progressions by music theory

def get_finetune_token():
    chord_progressions = ['F:maj7', 'E:min7', 'D:min7', 'C:maj(2)', 'Bb:9(#11)',
                          'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min',
                          'F:maj', 'E:min7', 'A:7', 'D:min7', 'C#:min7']
    cprog_augmented = augment_chord_list(chord_progressions,
                                         selected_prog=[['A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7']],
                                         # replace=[['A:min', 'B:min', 'E:min7', 'A:min']],
                                         strategy='select-gen',
                                         n_candidate=5,
                                         n_recurse=15
                                         )

    chordnum = sum([len(x) for x in cprog_augmented])
    print(chordnum)

    data = []
    for i, e in enumerate(cprog_augmented):
        data.extend(mod_and_tokenize_chords(e))

    data = data
    random.shuffle(data)
    out = []
    for i in data:
        out.extend(i)
        out.append('[cls]')
    return out


def generate_data():
    tokens = get_finetune_token()

    chars = sorted(list(set(tokens)))
    vocab_size = len(chars)
    print("all the unique characters:", ', '.join(chars))
    print(f"fine-tuning dataset vocab size: {vocab_size:,}")

    stoi, itos = get_vocab_dict(basedata_path)

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
    shutil.copyfile(basedata_path, './meta.pkl')


if __name__ == '__main__':
    generate_data()
    # out = mod_and_tokenize_chords(c_ws)
    # print(out)

'''
5588
all the unique characters: #11, #5, #9, *3, *5, ., /, 11, 2, 3, 5, 7, 9, A, Ab, B, Bb, C, C#, Cb, D, Db, E, Eb, F, F#, Fb, G, G#, Gb, [cls], add, aug, b3, b5, b7, b9, dim, dim7, hdim7, maj, maj6, maj7, min, min6, min7, min9, sus4
fine-tuning dataset vocab size: 48
train has 305,020 tokens
val has 33,892 tokens
'''
