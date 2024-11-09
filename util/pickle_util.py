import pickle


def get_pdata(fpath):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    return data


def get_vocab_dict(metapath):
    """
    Return stoi and itos
    :param metapath:
    :return:
    """
    data = get_pdata(metapath)
    return data['stoi'], data['itos']


if __name__ == '__main__':
    data = get_pdata('../data/chords-ws/meta.pkl')
    print(data)
