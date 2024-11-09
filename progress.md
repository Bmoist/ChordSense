# Record of Progress

### Apr.3

- Trained a 10M base model
    - nlayer = 6, nhead = 6, emb = 384
    - vocab = 75
    - train / val:  73,896,826 / 8,210,759 tokens
- Fine-tuned it on an artificial, flawed dataset
    - 3,710,250 / 412,250 tokens, mostly repetitive
    - P(hdim & hdim) did increase.
    - The second hdim's bass is also correctly captured
    - However, the chord relation is not correct (or very rarely)
        - Output: Bbhdim7.Ehdim7add3/3
        - Should be: Bbhdim, Ghdim

- TODO:
    - Tuning model hyperparameters
    - Looking into the embedding space
    - Encoder model?
        - (for example, write the chord relationship out, so GPT knows that the two hdim are a major third apart?)
        - Or is this really stupid, and generally can be solved by increasing model size?
    - Ask TA