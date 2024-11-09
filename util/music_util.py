from harte.harte import Harte
import datetime

pitch_names = ['Fb', 'Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F',
               'C',
               'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#']

pitch_name_scale = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

Harte_intv = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
Harte_root = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'N']
Harte_acc = ['#', 'b']

## Chords
c_wow = ['Eb:maj', 'Eb:maj/7', 'C:min', 'Eb:maj7/5', 'F:min7/b3', 'Bb:maj', 'Eb:maj',
         'G:min7', 'C:min7', 'G:min7', 'C:min7', 'Bb:7',
         'Eb:maj', 'Eb:maj/7', 'C:min', 'Eb:maj7/5', 'F:min7/b3', 'G:maj/5', 'C:maj/3',
         'F:maj7', 'F:sus2/3', 'C:maj/5', 'G:7', 'C:sus4', 'C:maj', 'Ab:maj/2', 'Bb:maj',
         'Eb:maj', 'Eb:maj/3', 'Ab:maj', 'G:hdim7', 'F:min/5', 'C:7', 'F:sus4', 'F:min',
         'F:min', 'B:dim7', 'Eb:maj(2)/3', 'D:dim/b5',
         'Eb:maj/3', 'D:hdim7/b7', 'Bb:7', 'Eb:aug', 'C:min',
         'F:min(9)', 'F:min', 'Eb:maj/3', 'F:min7/b3', 'Ab:maj9/3',
         'Eb:maj(2)/5', 'Bb:maj', 'F:min7(4)/5', 'Ab:maj/3',
         'Eb:maj/5', 'Ab:maj/2', 'Fb:min/4', 'Bb:7', 'Eb:maj']

c_wow_plus = ['Eb:maj', 'F:min7/b7', 'Eb:maj', 'F:hdim7/b7', 'Db:maj/2',
              'Eb:maj', 'Ab:maj', 'D:maj/3', 'Eb:maj/5', 'G:min', 'C:min7', 'Ab:maj', 'F:min7', 'Bb:min7',
              'Eb:maj', 'Db:maj', 'C:min', 'Db:7/b7']

c_ws = ['F:maj7', 'E:min7', 'D:min7', 'C:maj(2)', 'Bb:9(#11)',
        'F:maj7', 'E:min7', 'A:min', 'F#:hdim7', 'D:hdim7/b3', 'E:min7', 'A:min',
        'F:maj', 'E:min7', 'A:7', 'D:min7', 'C#:min7']

c_net1 = ['F:maj7', 'E:7', 'A:min7/5', 'Ab:min7/5', 'G:min7/5', 'G:dim7', 'F:maj7']

pitch_map = {
    'B#': 0,
    'C': 0,
    'C#': 1,
    'Db': 1,
    'D': 2,
    'D#': 3,
    'Eb': 3,
    'E': 4,
    'Fb': 4,
    'E#': 5,
    'F': 5,
    'F#': 6,
    'Gb': 6,
    'G': 7,
    'G#': 8,
    'Ab': 8,
    'A': 9,
    'A#': 10,
    'Bb': 10,
    'B': 11,
    'Cb': 11,
}


def encode_chord_type_loose(type):
    if 'min' in type and 'minmaj' not in type:
        return 'min'
    elif 'maj' in type:
        return 'maj'
    elif 'hdim' in type:
        return 'hdim'
    elif 'dim' in type:
        return 'dim'
    else:
        return type


def encode_chord_type_strict(type):
    return type


def encode_chord_pg(progression, enc=encode_chord_type_strict):
    """
    input a chord progression, converts to pitch_class, chord_type_enc
    :param enc:
    :param progression:
    :return:
    """
    out = []
    for i, e in enumerate(progression):
        if e == '[cls]':
            out.append([0, '[cls]'])
        elif len(e) != 2:
            print("Skip Abnormal chord!", e)
            continue
        else:
            try:
                out.append([pitch_map[e[0]], enc(e[1])])
            except:
                print("Skip flawed chord!: ", e)
                continue
    return out


def search_enc_chord_pg(src, target, ignore_cls=False):
    """
    search all occurrences of a target chord progression
    :param src: encoded list to be searched at
    :param target: encoded target chord progression
    :return: list of indices where the progression starts
    """
    out = []
    if type(src[0][0]) != int:
        raise Exception("Expect encoded list of chords, got: {}", format(src[0]))

    if ignore_cls:
        for e in target:
            if '[cls]' in e:
                return []

    def check_seq(search_seq):
        if search_seq[0][1] != target[0][1]:
            return False
        mod_idx = search_seq[0][0] - target[0][0]
        for j in range(len(target)):
            if (search_seq[j][0] - mod_idx + 12) % 12 != target[j][0]:
                return False
            elif search_seq[j][1] != target[j][1]:
                return False
        return True

    for i in range(len(src)):
        if i > len(src) - len(target):
            break  # not enough space to search
        if check_seq(src[i:i + len(target)]):
            out.append(i)
    return out


def search_chord_pg(src, target):
    """
    input a list of chord symbols like: [E, maj]
    :param src:
    :param target:
    :return:
    """
    src_enc = encode_chord_pg(src)
    target_enc = encode_chord_pg(target)
    return search_enc_chord_pg(src_enc, target_enc)


def standardize_pitch_name(name):
    if name in pitch_names:
        return name
    elif name == 'N' or name == 'X':
        return 'N'
    else:
        root = name[0]
        n_mods = len(name) - 1
        if 'b' in name:
            assert '#' not in name
            new_idx = (pitch_name_scale.index(root) - n_mods) % len(pitch_name_scale)
            return pitch_name_scale[new_idx]
        elif '#' in name:
            assert 'b' not in name
            new_idx = (pitch_name_scale.index(root) + n_mods) % len(pitch_name_scale)
            return pitch_name_scale[new_idx]
        else:
            raise Exception("Unknown accidental notation")


def format_chord_symbol_preliminary(symbol):
    if ":" not in symbol:
        if symbol == 'N' or symbol == 'X':
            return 'N'
        elif '/' in symbol:
            tmp = symbol.rsplit('/')
            return tmp[0] + ':maj/' + tmp[1]
        else:
            return symbol + ':maj'
    else:
        return symbol


def transpose_chord_seq(chord_list):
    """
    @param chord_list: list of Harte notations
    """
    assert type(chord_list) == list
    mod_chord_seq = []
    for i in range(17):
        # for i in range(1):
        mod_chords = []
        # Loop through all the 16 keys (including enharmonics)
        for e in chord_list:
            if e == '[cls]':
                raise Exception("No cls!")
            symbol = format_chord_symbol_preliminary(e)
            # symbol = Harte(e).prettify()
            if symbol == 'N':
                continue  # skip N for now
                # mod_chords.append(e)
            elif ':' not in symbol:
                raise Exception("Unknown chord symbol: {}!".format(symbol))
            root = symbol.split(':')[0]
            mod_root = modulate_pitch(root, i)
            mod_chords.append(mod_root + ':' + symbol.split(':')[-1])
        mod_chord_seq.append(mod_chords)
    return mod_chord_seq


def modulate_pitch(pitch_name, k):
    assert k >= 0
    # Find the index of the input pitch in the circle
    pitch_name = standardize_pitch_name(pitch_name)
    try:
        pitch_index = pitch_names.index(pitch_name)
    except ValueError:
        raise ValueError(
            "Invalid pitch name {}. Please provide a valid pitch name from the circle of fifths.".format(pitch_name))

    # Calculate the new index after modulation
    if pitch_index + k >= len(pitch_names):
        pitch_index += 5 * ((pitch_index + k) // 12)  # a very nasty hotfix to fix transposition by circle of fifth
    new_index = (pitch_index + k) % len(pitch_names)

    # Return the modulated pitch name
    return pitch_names[new_index]


class ChordParserFSM:
    def __init__(self):
        self.states = {
            "root": self.root_state,
            "quality": self.q_state,
            "add": self.add_state,
            "bass": self.bass_state,

            "error": self.error_state
        }
        self.current_state = "root"
        self.result = {
            "root": "",
            "quality": "",
            "add": [],
            "bass": ""
        }

    def reset(self):
        self.current_state = 'root'
        self.result = {
            "root": "",
            "quality": "",
            "add": [],
            "bass": ""
        }

    def parse(self, chord_symbol):
        try:
            sym = Harte(chord_symbol).prettify()
        except Exception as e:
            raise Exception("Cannot parse chord: {}. {}".format(chord_symbol, e))
        for char in sym:
            self.states[self.current_state](char)
        if self.current_state == "error":
            return None
        else:
            return self.result

    def get_token(self, k):
        if k == 'root' or k == 'quality':
            return self.result[k]
        elif k == 'bass':
            if self.result[k] == '':
                return ''
            else:
                return ['/', self.result[k]]
        elif k == 'add':
            while '' in self.result[k]:
                self.result[k].remove('')
            res = ''
            if len(self.result[k]) != 0:
                res = ['add']
                res.extend(self.result[k])
            return res
        else:
            if k not in self.result.keys():
                raise Exception("Unimplemented key")
            return ''

    def tokenize(self, chord_symbol):
        result = []
        self.parse(chord_symbol)
        for k in self.result.keys():
            token = self.get_token(k)
            if token == "#":
                print(chord_symbol)
            if token == '':
                continue
            if type(token) == list:
                result.extend(token)
            else:
                result.append(token)
        return result

    def root_state(self, char):
        if char in Harte_root:
            self.result["root"] += char
        elif char in Harte_acc:
            self.result['root'] += char
        elif char == ":":
            self.current_state = "quality"
        elif char == '(':
            self.current_state = 'add_note'
        else:
            self.states["error"]('cannot parse: ' + char)

    def q_state(self, char):
        if char.isalpha() or char in Harte_intv:
            self.result["quality"] += char
        elif char == "/":
            self.current_state = "bass"
        elif char == '(':
            self.current_state = 'add'
        else:
            self.states["error"](char)

    def add_state(self, char):
        if char in Harte_intv or char in Harte_acc or char == "*":
            # combine *11, #9 etc. into one token
            if len(self.result['add']) == 0:
                self.result['add'].append(char)
            else:
                self.result['add'][-1] = self.result['add'][-1] + char
        elif char == ',' or char == ')' or char == ' ':
            self.result['add'].append('')
        elif char == "/":
            self.current_state = "bass"
        else:
            self.states['error']('error during add_state:' + char)

    def bass_state(self, char):
        if char in Harte_acc or char in Harte_intv:
            self.result["bass"] += char
        elif char == '(':
            self.current_state = 'add'
        else:
            self.states["error"](char)

    def error_state(self, char):
        raise ValueError(f"Unexpected character: {char}")


def test_modulate_pitch():
    for name in pitch_names:
        for i in range(17):
            assert (pitch_map[modulate_pitch(name, i)] + i * 5) % 12 == pitch_map[name]


def convert_to_harte(token_list):
    """
    Crappy buggy nasty converter
    :param token_list:
    :return:
    """
    ctk = ''  # current token
    out = []

    track_adds = False
    for i, e in enumerate(token_list):
        if ',' in e:
            raise Exception("Encountered , in token list!")
        if e == '[cls]':
            track_adds = False
            if ctk and '(' in ctk and ')' not in ctk:
                ctk += ')'
                out.append(ctk)
            out.append(e)
        elif e == '.' and ctk:
            track_adds = False
            if ctk and '(' in ctk and ')' not in ctk:
                ctk += ')'
            out.append(ctk)
            ctk = ''
        elif e in pitch_names:
            ctk += (e + ':')
        elif e == '/':
            track_adds = False
            if '(' in ctk and ')' not in ctk:
                ctk += ')'
            ctk += e
        elif track_adds:
            if i > 0 and token_list[i - 1] == 'add':
                ctk += e
            else:
                ctk += (',' + e)
        elif e == 'add':
            track_adds = True
            ctk += '('
        elif ',' not in e and e != '.':
            ctk += e
    if ctk:
        if '(' in ctk and ')' not in ctk:
            ctk += ')'
        out.append(ctk)

    return out


if __name__ == '__main__':
    # print(standardize_pitch_name('Cbb'))
    # test_modulate_pitch()

    print(convert_to_harte(['.', 'A', 'maj', '.', '.', 'B', 'min']))
