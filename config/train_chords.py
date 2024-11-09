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
Train the model on tokenized Choco Dataset.
all the unique characters: #1, #11, #4, #9, *3, *5, *b3, ., /, 11, 2, 3, 4, 5, 6, 7, 9, 
A, A#, Ab, B, Bb, C, C#, C##, D, D#, Db, E, Eb, F, F#, F##, G, G#, G##, Gb, N, [cls], 
add, aug, b13, b3, b4, b5, b6, b7, b9, bb6, dim, dim7, hdim7, maj, maj6, maj7, maj9, 
min, min6, min7, min9, sus2, sus4

vocab size: 62
train has 3,731,523 tokens
val has 414,614 tokens
"""

out_dir = 'out-chords'
# init_from = 'resume'
eval_interval = 250
eval_iters = 200
log_interval = 10  # don't print too too often

# Similar vocab size, over 30 times bigger than the Shakespear char dataset.
# Maybe still overfit, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = 'chords'
wandb_run_name = 'mini-gpt'

dataset = 'chords'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 12000
lr_decay_iters = 12000  # make equal to max_iters usually
min_lr = 1e-3  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
device = 'mps'  # Mac m1 acceleration
compile = False  # do not torch compile the model
