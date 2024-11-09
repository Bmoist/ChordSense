import time

# out_dir = 'out-chords-oworld'
out_dir = 'out-chords-ws-selectgen'
dataset = 'chords-ws-selectgen'
init_from = 'resume'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters
eval_interval = 25
eval_iters = 40
log_interval = 5

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

wandb_log = False  # feel free to turn on
# wandb_project = 'chords-ws'
# wandb_run_name = 'ft-' + str(time.time())

# Nano gpt example:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
# dataset = 'chords-ft'

# finetune at constant LR
learning_rate = 3e-5
max_iters = 2900
decay_lr = False

device = 'mps'  # Mac m1 acceleration
compile = False  # do not torch compile the model
