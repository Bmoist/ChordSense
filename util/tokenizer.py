"""
Tokenize according to Chordinator's method
'.' 'C' 'm' 'add' '#7' '.'

TODO:
[√] mark the start of each piece using a cls token
[√] create a FSM to decode Harte chord symbol into elements
    wrapped by a [.] token representing the tuples
[√] Modulate to all 12 keys for data augmentation
[√] shuffle the data! so that fewer pieces from the same data source are put together
"""

import pickle
import random
from datetime import datetime

import jams
import numpy as np
import os
from util.music_util import transpose_chord_seq, ChordParserFSM
from harte.harte import Harte

jam_dir = '/Users/kurono/Desktop/v1.0.0/jams'
books = ['jaah', 'biab-internet-corpus', 'schubert-winterreise-audio', 'billboard', 'weimar', 'robbie-williams',
         'wikifonia', 'nottingham', 'isophonics', 'rock-corpus', 'rwc-pop', 'real-book', 'ireal-pro', 'jazz-corpus',
         'when-in-rome', 'uspop2002', 'schubert-winterreise-score', 'mozart-piano-sonatas', 'casd']
random.seed(0)


def get_jam_data():
    jam_data = dict()
    for i in books:
        jam_data[i] = []
    files = []
    for f in os.listdir(jam_dir):
        files.append(f)
    files.sort()
    for f in files:
        key = f.rsplit('_', 1)[0]
        if key not in books:
            raise Exception("Unknown key")
        obj = jams.load(jam_dir + '/' + f, validate=False, strict=False)
        jam_data[key].append(obj)
    return jam_data


def get_jam_chord(jam_piece, dont_search_chord_harte=False):
    chords = []
    chord_ann = jam_piece.annotations.search(namespace="chord")
    if len(chord_ann) > 1:
        if dont_search_chord_harte:
            chord_ann = chord_ann[0]
        else:
            chord_ann = chord_ann.search(namespace='chord_harte')[0]
    else:
        chord_ann = chord_ann[0]
    for i in chord_ann.data:
        chords.append(i.value)
    return chords


def tokenize_chord_list(clist):
    """
    Input a list of chords (Harte notation, return a sequence of parsed tokens
    :param clist:
    :return:
    """
    output = []
    chord_parser = ChordParserFSM()
    for c in clist:
        chord_parser.reset()
        output.extend(chord_parser.tokenize(c))
        output.append('.')  # separation token
    return output


def mod_and_tokenize_chords(clist):
    """
    Input a list of chords, return tokenized list of modulated chord sequences
    :param clist:
    :return:
    """
    mod_chord_list = transpose_chord_seq(clist)
    out = []
    for mc in mod_chord_list:
        out.append(tokenize_chord_list(mc))
    return out


def parse_jam_chord(jam_piece, dont_search_chord_harte=False):
    """
    @param chord_id: if it is none, use 0
    :return: list of tokens (each corresponds to a tonality)
    """
    try:
        harte_chord_list = get_jam_chord(jam_piece, dont_search_chord_harte)
        trim_flawed_chords(harte_chord_list)
        mod_chord_list = transpose_chord_seq(harte_chord_list)
    except Exception as e:
        print("Encountered exception while parsing piece. E: {}".format(e))
        return []

    token_list = []
    for mod_c in mod_chord_list:
        tmp_clist = tokenize_chord_list(mod_c)
        token_list.append(tmp_clist)
    return token_list


def trim_flawed_chords(chord_list):
    """
    Especially for Biab dataset. Wtf is bass = bb1, bb9?
    - Remove all bass if a flawed notation is found
    :param chord_list:
    """
    reformat = False
    for e in chord_list:
        if 'bb' in e or 'b1' in e:
            reformat = True
            break
    if not reformat:
        return reformat
    for i, e in enumerate(chord_list):
        chord_list[i] = chord_list[i].rsplit('/')[0]
    return reformat


def shuffle_data(data):
    print("Total", len(data), "pieces")
    random.shuffle(data)
    return data


def tokenize_dataset():
    """
    Return a sequence of tokens, each sheet separated by a [cls] token
    :return:
    """
    jam_data = get_jam_data()
    # test tokenizer:
    test_tokens = []

    for i in range(len(books)):
        # for i in range(0, 1):
        print("Parsing:", books[i], 'i:', i, 'size: ', len(jam_data[books[i]]))
        # chord_num = 0
        for j in range(len(jam_data[books[i]])):
            # for j in range(min(10, len(jam_data[books[i]]))):
            if j % 200 == 0:
                print("- parsing", j, 'th piece')
            # print(datetime.now())
            # Hotfix for books[18] - data namespace flawed
            # chord_num += len(get_jam_chord(jam_data[books[i]][j], True if i == 18 else False))
            result = parse_jam_chord(jam_data[books[i]][j], True if i == 18 else False)
            test_tokens.extend(result)
        # print("Chord num for book", books[i], chord_num)

    test_tokens = shuffle_data(test_tokens)
    print("Total {} pieces after modulation".format(len(test_tokens)))
    token_seq = []
    for i, e in enumerate(test_tokens):
        token_seq.extend(e)
        if i < len(test_tokens) - 1:
            token_seq.append('[cls]')
    return token_seq


def tokenize_specific_dataset(book_name):
    """
    Return a sequence of tokens, each sheet separated by a [cls] token
    :return:
    """
    assert book_name in books
    jam_data = get_jam_data()
    # test tokenizer:
    test_tokens = []

    print("Parsing:", book_name, 'size: ', len(jam_data[book_name]))
    # chord_num = 0
    for j in range(len(jam_data[book_name])):
        # for j in range(min(10, len(jam_data[books[i]]))):
        if j % 200 == 0:
            print("- parsing", j, 'th piece')
        # print(datetime.now())
        # Hotfix for books[18] - data namespace flawed
        # chord_num += len(get_jam_chord(jam_data[books[i]][j], True if i == 18 else False))
        result = parse_jam_chord(jam_data[book_name][j], True if book_name == 'casd' else False)
        test_tokens.extend(result)
    # print("Chord num for book", books[i], chord_num)

    test_tokens = shuffle_data(test_tokens)
    print("Total {} pieces after modulation".format(len(test_tokens)))
    token_seq = []
    for i, e in enumerate(test_tokens):
        token_seq.extend(e)
        if i < len(test_tokens) - 1:
            token_seq.append('[cls]')
    return token_seq


def get_all_chords():
    jam_data = get_jam_data()
    # test tokenizer:
    test_tokens = []
    for i in range(len(books)):
        # for i in range(1, 2):
        print("Parsing:", books[i], 'i:', i, 'size: ', len(jam_data[books[i]]))


if __name__ == '__main__':
    cp = ChordParserFSM()
    print(cp.tokenize("Bb:hdim7/b5"))
