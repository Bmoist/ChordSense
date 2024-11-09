import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

"""
all the unique characters: ##1, ##4, #1, #11, #13, #3, #4, #5, #6, #7, #9, *1, *11, *3, *5, *b3, ., /, 1, 11, 13, 2, 3, 4, 5, 6, 7, 9, A, Ab, B, Bb, C, C#, Cb, D, Db, E, Eb, F, F#, Fb, G, G#, Gb, [cls], add, aug, b1, b13, b2, b3, b4, b5, b6, b7, b8, b9, bb3, bb7, bb9, dim, dim7, hdim7, maj, maj6, maj7, maj9, min, min6, min7, min9, minmaj7, sus2, sus4
vocab size: 75
train has 73,896,826 tokens
val has 8,210,759 tokens

"""

out_dir = 'out-revchords'
eval_interval = 250
eval_iters = 200
log_interval = 10  # don't print too too often

# Similar vocab size, over 30 times bigger than the Shakespear char dataset.
# Maybe still overfit, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
# wandb_project = 'chords'
# wandb_run_name = 'mini-gpt'

dataset = 'revchords'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
device = 'mps'  # Mac m1 acceleration
compile = False  # do not torch compile the model
