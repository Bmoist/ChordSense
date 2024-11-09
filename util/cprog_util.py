from util.tokenizer import tokenize_chord_list
from util.model_util import gen_chord_context, gen_chord_context_with_replace, get_model, filter_chord_context
from util.music_util import convert_to_harte


def get_chord_pg(tokens):
    """
    input a list of Harte chords, decodes into chord progressions (ignoring adds and bass)
    :return:
    """
    watch_count = 2
    out = []
    tmp = []
    for i, e in enumerate(tokens):
        if e == '.' or e == '[cls]':
            watch_count = 2  # pay attention to the following two tokens
            if len(tmp) < 2 and e != '[cls]':
                print("Skipping abnormal chord", tmp)
            else:
                out.append(tmp)
            tmp = []
        if watch_count > 0 and e != '.' and e != '[cls]':
            tmp.append(e)
            watch_count -= 1
    while [] in out:
        out.remove([])
    return out


def get_harte_chord_pg(clist):
    tokens = tokenize_chord_list(clist)
    return get_chord_pg(tokens)


def get_sublist_idx(base_list, target_list):
    indices = []
    for i in range(len(base_list) - len(target_list)):
        is_sublist = True
        for j in range(len(target_list)):
            if base_list[i + j] != target_list[j]:
                is_sublist = False
                break
        if is_sublist:
            indices.append(i)
    return indices


def obtain_list_context(base_list, target_list):
    indice = get_sublist_idx(base_list, target_list)
    if not indice:
        print("List:", target_list, "not in base list")
        return []
    idx = indice[0]
    n_ctx = 100
    out = []
    i_start, i_end = idx, idx + len(target_list)
    for i in range(n_ctx):
        if i_start < 0:
            break
        out.append(base_list[i_start: i_end])
        i_start -= 1
    return out


def augment_chord_list(chord_prog, selected_prog=None, strategy=None, n_candidate=5, n_recurse=15, replace=None):
    """

    :param chord_prog:
    :param selected_prog:
    :param strategy: None, select-n-gram, select-gen, select-complete
    :return: tokenized chords
    """
    out = []
    if strategy is None:
        return [chord_prog] * 45
    elif strategy == 'select-n-gram':
        assert type(selected_prog) == list
        for prog in selected_prog:
            augmented = obtain_list_context(chord_prog, prog)
            if not augmented:
                print("Chord progression:", prog, "is not augmented!")
                continue
            out.extend(augmented * 13)
        # Now, add some of the original chord progression
        for i in range(int(len(out) * 0.1)):
            out.append(chord_prog)
    elif strategy == 'select-gen':
        # Generate context given chord sequence
        if replace:
            assert len(replace) == len(selected_prog)
        model, encode, decode = get_model('/Users/kurono/Documents/python/GEC/research/out-revchords')
        token_lists = [tokenize_chord_list(e) for e in selected_prog]
        replace_lists = [tokenize_chord_list(e) for e in replace] if replace else None
        context = []
        for i, cp in enumerate(token_lists):
            cp.reverse()
            if replace_lists:
                rep = replace_lists[i].copy()
                rep.reverse()
                out, _ = gen_chord_context_with_replace(cp, rep, model, encode, decode,
                                                        n_candidate=n_candidate,
                                                        n_recurse=n_recurse)
                result = filter_chord_context(out)
                context.extend(result)
            else:
                tmp_ctx = gen_chord_context(cp, model, encode, decode, n_candidate=n_candidate,
                                            n_recurse=n_recurse)
                result = filter_chord_context(tmp_ctx)
                context.extend(result)

        context = [convert_to_harte(e) for e in context]
        # Now, add some of the original chord progression
        for i in range(int(len(context) * 0.1)):
            context.append(chord_prog)
        return context

    elif strategy == 'select-complete':
        assert selected_prog is not None
        print("Warning: this is experimental with default value passed")
        # Hand-crafted list of progressions by music theory
        out = [['F:maj7', 'E:min7', 'D:min7', 'C:maj(2)', 'Bb:9(#11)',
                'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min',
                'F:maj', 'E:min7', 'A:7', 'D:min7', 'C#:min7'],
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
    else:
        raise Exception("Unknown chord strategy")

    return out
