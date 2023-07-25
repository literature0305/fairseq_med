#!/usr/bin/env bash

# evaluation
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 20 --remove-bpe
