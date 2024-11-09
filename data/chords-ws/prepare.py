import numpy as np
import os
import pickle
import random
import shutil

from util.tokenizer import mod_and_tokenize_chords
from util.pickle_util import get_vocab_dict
from util.cprog_util import augment_chord_list

basedata_path = '../chords/meta.pkl'

# Hand-crafted list of progressions by music theory
cprog_extra = [['F:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3'],
               ['C:maj', 'F:min', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj', 'C:maj', 'F:min', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj7', 'C:maj', 'F:min7', 'B:7', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['E:min', 'G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['E:min', 'A:min', 'G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7',
                'E:min7', 'A:min'],
               ['D:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['C:maj7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['C:maj7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['C:maj', 'F:maj7', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj', 'G:maj', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['G:7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['G:maj', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['A:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3'],
               ['C:maj', 'G:maj', 'E:min', 'A:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3',
                'E:min7', 'A:min'],
               ['F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['F:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3'],
               ['C:maj', 'F:min', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj', 'C:maj', 'F:min', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj7', 'C:maj', 'F:min7', 'B:7', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['E:min', 'G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['E:min', 'A:min', 'G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7',
                'E:min7', 'A:min'],
               ['D:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['C:maj7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['C:maj7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['C:maj', 'F:maj7', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj', 'G:maj', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['G:7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['G:maj', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['A:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3'],
               ['C:maj', 'G:maj', 'E:min', 'A:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3',
                'E:min7', 'A:min'],
               ['F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['F:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3'],
               ['C:maj', 'F:min', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj', 'C:maj', 'F:min', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj7', 'C:maj', 'F:min7', 'B:7', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['E:min', 'G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['E:min', 'A:min', 'G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7',
                'E:min7', 'A:min'],
               ['D:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['C:maj7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['C:maj7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['C:maj', 'F:maj7', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj', 'G:maj', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['G:7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['G:maj', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['A:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3'],
               ['C:maj', 'G:maj', 'E:min', 'A:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3',
                'E:min7', 'A:min'],
               ['F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['F:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3'],
               ['C:maj', 'F:min', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj', 'C:maj', 'F:min', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj7', 'C:maj', 'F:min7', 'B:7', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['E:min', 'G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['E:min', 'A:min', 'G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7',
                'E:min7', 'A:min'],
               ['D:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['C:maj7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['C:maj7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['C:maj', 'F:maj7', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj', 'G:maj', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['G:7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['G:maj', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['A:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3'],
               ['C:maj', 'G:maj', 'E:min', 'A:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3',
                'E:min7', 'A:min'],
               ['F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['F:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3'],
               ['C:maj', 'F:min', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj', 'C:maj', 'F:min', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj7', 'C:maj', 'F:min7', 'B:7', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['E:min', 'G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['E:min', 'A:min', 'G:7', 'F:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7',
                'E:min7', 'A:min'],
               ['D:min7', 'F:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['C:maj7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['C:maj7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['C:maj', 'F:maj7', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7'],
               ['C:maj', 'G:maj', 'B:maj', 'E:min', 'A:min', 'F#:hdim7', 'D:hdim7', 'E:min7', 'A:min'],
               ['G:7', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['G:maj', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min'],
               ['A:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3'],
               ['C:maj', 'G:maj', 'E:min', 'A:min', 'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3',
                'E:min7', 'A:min']]


def get_finetune_token():
    chord_progressions = ['F:maj7', 'E:min7', 'D:min7', 'C:maj(2)', 'Bb:9(#11)',
                          'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min',
                          'F:maj', 'E:min7', 'A:7', 'D:min7', 'C#:min7']
    cprog_augmented = augment_chord_list(chord_progressions,
                                         selected_prog=[['A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7']],
                                         strategy='select-n-gram')
    chordnum = sum([len(x) for x in cprog_augmented])
    print(chordnum)

    data = []
    for i in cprog_augmented:
        data.extend(mod_and_tokenize_chords(i))

    data = data * 100
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
