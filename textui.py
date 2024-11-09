from contextlib import nullcontext
from util.model_util import get_model, gen_next_chord, load_from_paths
from util.music_util import ChordParserFSM, convert_to_harte
import torch
import argparse

device = 'mps'


def get_user_chord_input():
    cp = ChordParserFSM()
    user_input = ''
    while True:
        try:
            user_input = input("Enter a harte chord symbol. Type [end] to stop.")
            if user_input == '[end]' or user_input == '[cls]':
                return [user_input]
            result = cp.tokenize(user_input)
            result.append('.')
            return result
        except:
            print("Invalid Harte chord symbol! Plz enter again.")
            continue
    return cp.tokenize(user_input)


def print_suggestion(chord, p):
    c = convert_to_harte(chord)[0]
    if p > 0.8:
        print("\x1B[31m{}\033[0m".format('Suggested chord: ' + c + ', Prob:' + str(p)))
    elif p > 0.5:
        print("\x1B[33m{}\033[0m".format('Suggested chord: ' + c + ', Prob:' + str(p)))
    elif p > 0.2:
        print("\x1B[34m{}\033[0m".format('Suggested chord: ' + c + ', Prob:' + str(p)))
    else:
        print('Suggested chord: ', c, 'Prob:', p)


def suggest_chord(model, encode, decode):
    """
    If the model gets excited (higher probability), the color changes...

    """
    user_input = get_user_chord_input()
    tok_seq = user_input
    with torch.no_grad():
        with nullcontext():
            while user_input[-1] != '[end]':
                print('tok_seq', tok_seq)
                chord, p = gen_next_chord(model, encode, decode, tok_seq)
                print_suggestion(chord, p)
                user_input = get_user_chord_input()
                tok_seq.extend(user_input)
    print("\n\n--------------------------")
    print("\x1B[34m[Info]\033[0m End of session. Your final chord sequence is:")
    return convert_to_harte(tok_seq)


def tui(ckpt_path, tokenizer_path, device):
    model, encode, decode = load_from_paths(ckpt_path=ckpt_path, tokenizer_path=tokenizer_path, device_=device)
    print(suggest_chord(model, encode, decode))


def main():
    parser = argparse.ArgumentParser(
        description='Text-based User Interface to interact with pretrained and fine-tuned chord models'
    )
    parser.add_argument('--ckpt_path', type=str, required=False, help="Path to the torch model checkpoint")
    parser.add_argument('--tokenizer_path', type=str, required=False, help="Path to the chord tokenizer")
    parser.add_argument('--device', type=str, required=False, help="gpu / mps / cpu")
    args = parser.parse_args()
    tui(ckpt_path=args.ckpt_path, tokenizer_path=args.tokenizer_path, device=args.device)


def test_main():
    ckpt_path = '/Users/kurono/Documents/python/GEC/ChordSense/out-chords-ws-selectgen/ckpt.pt'
    tokenizer_path = '/Users/kurono/Documents/python/GEC/ChordSense/data/chords-ws-selectgen/meta.pkl'
    device = 'mps'
    tui(ckpt_path, tokenizer_path, device)


if __name__ == '__main__':
    main()
    # test_main()
